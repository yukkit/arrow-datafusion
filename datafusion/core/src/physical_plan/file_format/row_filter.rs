// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{Array, BooleanArray};
use arrow::datatypes::{DataType, Schema};
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow::record_batch::RecordBatch;
use datafusion_common::{Column, DataFusionError, Result, ScalarValue, ToDFSchema};
use datafusion_expr::expr_rewriter::{ExprRewritable, ExprRewriter, RewriteRecursion};

use datafusion_expr::Expr;
use datafusion_optimizer::utils::split_conjunction_owned;
use datafusion_physical_expr::execution_props::ExecutionProps;
use datafusion_physical_expr::{create_physical_expr, PhysicalExpr};
use parquet::arrow::arrow_reader::{ArrowPredicate, RowFilter};
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::ParquetMetaData;
use std::sync::Arc;

use crate::physical_plan::metrics;

/// This module contains utilities for enabling the pushdown of DataFusion filter predicates (which
/// can be any DataFusion `Expr` that evaluates to a `BooleanArray`) to the parquet decoder level in `arrow-rs`.
/// DataFusion will use a `ParquetRecordBatchStream` to read data from parquet into arrow `RecordBatch`es.
/// When constructing the  `ParquetRecordBatchStream` you can provide a `RowFilter` which is itself just a vector
/// of `Box<dyn ArrowPredicate>`. During decoding, the predicates are evaluated to generate a mask which is used
/// to avoid decoding rows in projected columns which are not selected which can significantly reduce the amount
/// of compute required for decoding.
///
/// Since the predicates are applied serially in the order defined in the `RowFilter`, the optimal ordering
/// will depend on the exact filters. The best filters to execute first have two properties:
///     1. The are relatively inexpensive to evaluate (e.g. they read column chunks which are relatively small)
///     2. They filter a lot of rows, reducing the amount of decoding required for subsequent filters and projected columns
///
/// Given the metadata exposed by parquet, the selectivity of filters is not easy to estimate so the heuristics we use here primarily
/// focus on the evaluation cost.
///
/// The basic algorithm for constructing the `RowFilter` is as follows
///     1. Recursively break conjunctions into separate predicates. An expression like `a = 1 AND (b = 2 AND c = 3)` would be
///        separated into the expressions `a = 1`, `b = 2`, and `c = 3`.
///     2. Determine whether each predicate is suitable as an `ArrowPredicate`. As long as the predicate does not reference any projected columns
///        or columns with non-primitive types, then it is considered suitable.
///     3. Determine, for each predicate, the total compressed size of all columns required to evaluate the predicate.
///     4. Determine, for each predicate, whether all columns required to evaluate the expression are sorted.
///     5. Re-order the predicate by total size (from step 3).
///     6. Partition the predicates according to whether they are sorted (from step 4)
///     7. "Compile" each predicate `Expr` to a `DatafusionArrowPredicate`.
///     8. Build the `RowFilter` with the sorted predicates followed by the unsorted predicates. Within each partition
///        the predicates will still be sorted by size.

/// A predicate which can be passed to `ParquetRecordBatchStream` to perform row-level
/// filtering during parquet decoding.
#[derive(Debug)]
pub(crate) struct DatafusionArrowPredicate {
    physical_expr: Arc<dyn PhysicalExpr>,
    projection_mask: ProjectionMask,
    projection: Vec<usize>,
    /// how many rows were filtered out by this predicate
    rows_filtered: metrics::Count,
    /// how long was spent evaluating this predicate
    time: metrics::Time,
}

impl DatafusionArrowPredicate {
    pub fn try_new(
        candidate: FilterCandidate,
        schema: &Schema,
        metadata: &ParquetMetaData,
        rows_filtered: metrics::Count,
        time: metrics::Time,
    ) -> Result<Self> {
        let props = ExecutionProps::default();

        let schema = schema.project(&candidate.projection)?;
        let df_schema = schema.clone().to_dfschema()?;

        let physical_expr =
            create_physical_expr(&candidate.expr, &df_schema, &schema, &props)?;

        // ArrowPredicate::evaluate is passed columns in the order they appear in the file
        // If the predicate has multiple columns, we therefore must project the columns based
        // on the order they appear in the file
        let projection = match candidate.projection.len() {
            0 | 1 => vec![],
            _ => remap_projection(&candidate.projection),
        };

        Ok(Self {
            physical_expr,
            projection,
            projection_mask: ProjectionMask::roots(
                metadata.file_metadata().schema_descr(),
                candidate.projection,
            ),
            rows_filtered,
            time,
        })
    }
}

impl ArrowPredicate for DatafusionArrowPredicate {
    fn projection(&self) -> &ProjectionMask {
        &self.projection_mask
    }

    fn evaluate(&mut self, batch: RecordBatch) -> ArrowResult<BooleanArray> {
        let batch = match self.projection.is_empty() {
            true => batch,
            false => batch.project(&self.projection)?,
        };

        // scoped timer updates on drop
        let mut timer = self.time.timer();
        match self
            .physical_expr
            .evaluate(&batch)
            .map(|v| v.into_array(batch.num_rows()))
        {
            Ok(array) => {
                if let Some(mask) = array.as_any().downcast_ref::<BooleanArray>() {
                    let bool_arr = BooleanArray::from(mask.data().clone());
                    let num_filtered = bool_arr.len() - true_count(&bool_arr);
                    self.rows_filtered.add(num_filtered);
                    timer.stop();
                    Ok(bool_arr)
                } else {
                    Err(ArrowError::ComputeError(
                        "Unexpected result of predicate evaluation, expected BooleanArray".to_owned(),
                    ))
                }
            }
            Err(e) => Err(ArrowError::ComputeError(format!(
                "Error evaluating filter predicate: {:?}",
                e
            ))),
        }
    }
}

/// Return the number of non null true vaulues in an array
// TODO remove when https://github.com/apache/arrow-rs/issues/2963 is released
fn true_count(arr: &BooleanArray) -> usize {
    match arr.data().null_buffer() {
        Some(nulls) => {
            let null_chunks = nulls.bit_chunks(arr.offset(), arr.len());
            let value_chunks = arr.values().bit_chunks(arr.offset(), arr.len());
            null_chunks
                .iter()
                .zip(value_chunks.iter())
                .chain(std::iter::once((
                    null_chunks.remainder_bits(),
                    value_chunks.remainder_bits(),
                )))
                .map(|(a, b)| (a & b).count_ones() as usize)
                .sum()
        }
        None => arr.values().count_set_bits_offset(arr.offset(), arr.len()),
    }
}

/// A candidate expression for creating a `RowFilter` contains the
/// expression as well as data to estimate the cost of evaluating
/// the resulting expression.
pub(crate) struct FilterCandidate {
    expr: Expr,
    required_bytes: usize,
    can_use_index: bool,
    projection: Vec<usize>,
}

/// Helper to build a `FilterCandidate`. This will do several things
/// 1. Determine the columns required to evaluate the expression
/// 2. Calculate data required to estimate the cost of evaluating the filter
/// 3. Rewrite column expressions in the predicate which reference columns not in the particular file schema.
///    This is relevant in the case where we have determined the table schema by merging all individual file schemas
///    and any given file may or may not contain all columns in the merged schema. If a particular column is not present
///    we replace the column expression with a literal expression that produces a null value.
struct FilterCandidateBuilder<'a> {
    expr: Expr,
    file_schema: &'a Schema,
    table_schema: &'a Schema,
    required_column_indices: Vec<usize>,
    non_primitive_columns: bool,
    projected_columns: bool,
}

impl<'a> FilterCandidateBuilder<'a> {
    pub fn new(expr: Expr, file_schema: &'a Schema, table_schema: &'a Schema) -> Self {
        Self {
            expr,
            file_schema,
            table_schema,
            required_column_indices: vec![],
            non_primitive_columns: false,
            projected_columns: false,
        }
    }

    pub fn build(
        mut self,
        metadata: &ParquetMetaData,
    ) -> Result<Option<FilterCandidate>> {
        let expr = self.expr.clone();
        let expr = expr.rewrite(&mut self)?;

        if self.non_primitive_columns || self.projected_columns {
            Ok(None)
        } else {
            let required_bytes =
                size_of_columns(&self.required_column_indices, metadata)?;
            let can_use_index = columns_sorted(&self.required_column_indices, metadata)?;

            Ok(Some(FilterCandidate {
                expr,
                required_bytes,
                can_use_index,
                projection: self.required_column_indices,
            }))
        }
    }
}

impl<'a> ExprRewriter for FilterCandidateBuilder<'a> {
    fn pre_visit(&mut self, expr: &Expr) -> Result<RewriteRecursion> {
        if let Expr::Column(column) = expr {
            if let Ok(idx) = self.file_schema.index_of(&column.name) {
                self.required_column_indices.push(idx);

                if DataType::is_nested(self.file_schema.field(idx).data_type()) {
                    self.non_primitive_columns = true;
                }
            } else if self.table_schema.index_of(&column.name).is_err() {
                // If the column does not exist in the (un-projected) table schema then
                // it must be a projected column.
                self.projected_columns = true;
            }
        }
        Ok(RewriteRecursion::Continue)
    }

    fn mutate(&mut self, expr: Expr) -> Result<Expr> {
        if let Expr::Column(Column { name, .. }) = &expr {
            if self.file_schema.field_with_name(name).is_err() {
                // the column expr must be in the table schema
                return match self.table_schema.field_with_name(name) {
                    Ok(field) => {
                        // return the null value corresponding to the data type
                        let null_value = ScalarValue::try_from(field.data_type())?;
                        Ok(Expr::Literal(null_value))
                    }
                    Err(e) => {
                        // If the column is not in the table schema, should throw the error
                        Err(DataFusionError::ArrowError(e))
                    }
                };
            }
        }

        Ok(expr)
    }
}

/// Computes the projection required to go from the file's schema order to the projected
/// order expected by this filter
///
/// Effectively this computes the rank of each element in `src`
fn remap_projection(src: &[usize]) -> Vec<usize> {
    let len = src.len();

    // Compute the column mapping from projected order to file order
    // i.e. the indices required to sort projected schema into the file schema
    //
    // e.g. projection: [5, 9, 0] -> [2, 0, 1]
    let mut sorted_indexes: Vec<_> = (0..len).collect();
    sorted_indexes.sort_unstable_by_key(|x| src[*x]);

    // Compute the mapping from schema order to projected order
    // i.e. the indices required to sort file schema into the projected schema
    //
    // Above we computed the order of the projected schema according to the file
    // schema, and so we can use this as the comparator
    //
    // e.g. sorted_indexes [2, 0, 1] -> [1, 2, 0]
    let mut projection: Vec<_> = (0..len).collect();
    projection.sort_unstable_by_key(|x| sorted_indexes[*x]);
    projection
}

/// Calculate the total compressed size of all `Column's required for
/// predicate `Expr`. This should represent the total amount of file IO
/// required to evaluate the predicate.
fn size_of_columns(columns: &[usize], metadata: &ParquetMetaData) -> Result<usize> {
    let mut total_size = 0;
    let row_groups = metadata.row_groups();
    for idx in columns {
        for rg in row_groups.iter() {
            total_size += rg.column(*idx).compressed_size() as usize;
        }
    }

    Ok(total_size)
}

/// For a given set of `Column`s required for predicate `Expr` determine whether all
/// columns are sorted. Sorted columns may be queried more efficiently in the presence of
/// a PageIndex.
fn columns_sorted(_columns: &[usize], _metadata: &ParquetMetaData) -> Result<bool> {
    // TODO How do we know this?
    Ok(false)
}

/// Build a [`RowFilter`] from the given predicate `Expr`
pub fn build_row_filter(
    expr: Expr,
    file_schema: &Schema,
    table_schema: &Schema,
    metadata: &ParquetMetaData,
    reorder_predicates: bool,
    rows_filtered: &metrics::Count,
    time: &metrics::Time,
) -> Result<Option<RowFilter>> {
    let predicates = split_conjunction_owned(expr);

    let mut candidates: Vec<FilterCandidate> = predicates
        .into_iter()
        .flat_map(|expr| {
            if let Ok(candidate) =
                FilterCandidateBuilder::new(expr, file_schema, table_schema)
                    .build(metadata)
            {
                candidate
            } else {
                None
            }
        })
        .collect();

    if candidates.is_empty() {
        Ok(None)
    } else if reorder_predicates {
        candidates.sort_by_key(|c| c.required_bytes);

        let (indexed_candidates, other_candidates): (Vec<_>, Vec<_>) =
            candidates.into_iter().partition(|c| c.can_use_index);

        let mut filters: Vec<Box<dyn ArrowPredicate>> = vec![];

        for candidate in indexed_candidates {
            let filter = DatafusionArrowPredicate::try_new(
                candidate,
                file_schema,
                metadata,
                rows_filtered.clone(),
                time.clone(),
            )?;

            filters.push(Box::new(filter));
        }

        for candidate in other_candidates {
            let filter = DatafusionArrowPredicate::try_new(
                candidate,
                file_schema,
                metadata,
                rows_filtered.clone(),
                time.clone(),
            )?;

            filters.push(Box::new(filter));
        }

        Ok(Some(RowFilter::new(filters)))
    } else {
        let mut filters: Vec<Box<dyn ArrowPredicate>> = vec![];
        for candidate in candidates {
            let filter = DatafusionArrowPredicate::try_new(
                candidate,
                file_schema,
                metadata,
                rows_filtered.clone(),
                time.clone(),
            )?;

            filters.push(Box::new(filter));
        }

        Ok(Some(RowFilter::new(filters)))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::physical_plan::file_format::row_filter::FilterCandidateBuilder;
    use arrow::datatypes::Field;
    use datafusion_expr::{cast, col, lit};
    use parquet::arrow::parquet_to_arrow_schema;
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use rand::prelude::*;

    // Assume a column expression for a column not in the table schema is a projected column and ignore it
    #[test]
    #[should_panic(expected = "building candidate failed")]
    fn test_filter_candidate_builder_ignore_projected_columns() {
        let testdata = crate::test_util::parquet_test_data();
        let file = std::fs::File::open(&format!("{}/alltypes_plain.parquet", testdata))
            .expect("opening file");

        let reader = SerializedFileReader::new(file).expect("creating reader");

        let metadata = reader.metadata();

        let table_schema =
            parquet_to_arrow_schema(metadata.file_metadata().schema_descr(), None)
                .expect("parsing schema");

        let expr = col("projected_column").eq(lit("value"));

        let candidate = FilterCandidateBuilder::new(expr, &table_schema, &table_schema)
            .build(metadata)
            .expect("building candidate failed");

        assert!(candidate.is_none());
    }

    // We should ignore predicate that read non-primitive columns
    #[test]
    fn test_filter_candidate_builder_ignore_complex_types() {
        let testdata = crate::test_util::parquet_test_data();
        let file = std::fs::File::open(&format!("{}/list_columns.parquet", testdata))
            .expect("opening file");

        let reader = SerializedFileReader::new(file).expect("creating reader");

        let metadata = reader.metadata();

        let table_schema =
            parquet_to_arrow_schema(metadata.file_metadata().schema_descr(), None)
                .expect("parsing schema");

        let expr = col("int64_list").is_not_null();

        let candidate = FilterCandidateBuilder::new(expr, &table_schema, &table_schema)
            .build(metadata)
            .expect("building candidate");

        assert!(candidate.is_none());
    }

    // If a column exists in the table schema but not the file schema it should be rewritten to a null expression
    #[test]
    fn test_filter_candidate_builder_rewrite_missing_column() {
        let testdata = crate::test_util::parquet_test_data();
        let file = std::fs::File::open(&format!("{}/alltypes_plain.parquet", testdata))
            .expect("opening file");

        let reader = SerializedFileReader::new(file).expect("creating reader");

        let metadata = reader.metadata();

        let table_schema =
            parquet_to_arrow_schema(metadata.file_metadata().schema_descr(), None)
                .expect("parsing schema");

        let file_schema = Schema::new(vec![
            Field::new("bigint_col", DataType::Int64, true),
            Field::new("float_col", DataType::Float32, true),
        ]);

        // The parquet file with `file_schema` just has `bigint_col` and `float_col` column, and don't have the `int_col`
        let expr = col("bigint_col").eq(cast(col("int_col"), DataType::Int64));
        let expected_candidate_expr =
            col("bigint_col").eq(cast(lit(ScalarValue::Int32(None)), DataType::Int64));

        let candidate = FilterCandidateBuilder::new(expr, &file_schema, &table_schema)
            .build(metadata)
            .expect("building candidate");

        assert!(candidate.is_some());

        assert_eq!(candidate.unwrap().expr, expected_candidate_expr);
    }

    #[test]
    fn test_remap_projection() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            // A random selection of column indexes in arbitrary order
            let projection: Vec<_> = (0..100).map(|_| rng.gen()).collect();

            // File order is the projection sorted
            let mut file_order = projection.clone();
            file_order.sort_unstable();

            let remap = remap_projection(&projection);
            // Applying the remapped projection to the file order should yield the original
            let remapped: Vec<_> = remap.iter().map(|r| file_order[*r]).collect();
            assert_eq!(projection, remapped)
        }
    }
}
