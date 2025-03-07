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

use arrow::datatypes::{DataType, Schema};

use arrow::record_batch::RecordBatch;

use datafusion_common::{ColumnStatistics, DataFusionError, Result, ScalarValue};
use datafusion_expr::ColumnarValue;

use arrow::array::{make_array, Array, ArrayRef, BooleanArray, MutableArrayData};
use arrow::compute::{and_kleene, filter_record_batch, is_not_null, SlicesIterator};

use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

/// Expression that can be evaluated against a RecordBatch
/// A Physical expression knows its type, nullability and how to evaluate itself.
pub trait PhysicalExpr: Send + Sync + Display + Debug + PartialEq<dyn Any> {
    /// Returns the physical expression as [`Any`](std::any::Any) so that it can be
    /// downcast to a specific implementation.
    fn as_any(&self) -> &dyn Any;
    /// Get the data type of this expression, given the schema of the input
    fn data_type(&self, input_schema: &Schema) -> Result<DataType>;
    /// Determine whether this expression is nullable, given the schema of the input
    fn nullable(&self, input_schema: &Schema) -> Result<bool>;
    /// Evaluate an expression against a RecordBatch
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue>;
    /// Evaluate an expression against a RecordBatch after first applying a
    /// validity array
    fn evaluate_selection(
        &self,
        batch: &RecordBatch,
        selection: &BooleanArray,
    ) -> Result<ColumnarValue> {
        let tmp_batch = filter_record_batch(batch, selection)?;

        let tmp_result = self.evaluate(&tmp_batch)?;
        // All values from the `selection` filter are true.
        if batch.num_rows() == tmp_batch.num_rows() {
            return Ok(tmp_result);
        }
        if let ColumnarValue::Array(a) = tmp_result {
            let result = scatter(selection, a.as_ref())?;
            Ok(ColumnarValue::Array(result))
        } else {
            Ok(tmp_result)
        }
    }
    /// Return the expression statistics for this expression. This API is currently experimental.
    fn expr_stats(&self) -> Arc<dyn PhysicalExprStats> {
        Arc::new(BasicExpressionStats {})
    }
    /// Get a list of child PhysicalExpr that provide the input for this expr.
    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>>;

    /// Returns a new PhysicalExpr where all children were replaced by new exprs.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>>;
}

/// Statistics about the result of a single expression.
#[derive(Clone, Debug, PartialEq)]
pub struct ExprBoundaries {
    /// Maximum value this expression's result can have.
    pub max_value: ScalarValue,
    /// Minimum value this expression's result can have.
    pub min_value: ScalarValue,
    /// Maximum number of distinct values this expression can produce, if known.
    pub distinct_count: Option<usize>,
    /// Selectivity of this expression if it were used as a predicate, as a
    /// value between 0 and 1.
    pub selectivity: Option<f64>,
}

impl ExprBoundaries {
    /// Create a new `ExprBoundaries`.
    pub fn new(
        max_value: ScalarValue,
        min_value: ScalarValue,
        distinct_count: Option<usize>,
    ) -> Self {
        Self {
            max_value,
            min_value,
            distinct_count,
            selectivity: None,
        }
    }

    /// Try to reduce the expression boundaries to a single value if possible.
    pub fn reduce(&self) -> Option<ScalarValue> {
        if self.min_value == self.max_value {
            Some(self.min_value.clone())
        } else {
            None
        }
    }
}

/// A toolkit to work with physical expressions statistics. This API is currently experimental
/// and might be subject to change.
pub trait PhysicalExprStats: Send + Sync {
    /// Return an estimate about the boundaries of this expression's result would have (in
    /// terms of minimum and maximum values it can take as well the number of unique values
    /// it can produce). The inputs are the column-level statistics from the current physical
    /// plan.
    fn boundaries(&self, columns: &[ColumnStatistics]) -> Option<ExprBoundaries>;
}

#[derive(Debug, Clone)]
pub struct BasicExpressionStats {}

/// A dummy implementation of [`ExpressionStats`] that does not provide any statistics.
impl PhysicalExprStats for BasicExpressionStats {
    #[allow(unused_variables)]
    fn boundaries(&self, columns: &[ColumnStatistics]) -> Option<ExprBoundaries> {
        None
    }
}

/// Returns a copy of this expr if we change any child according to the pointer comparison.
/// The size of `children` must be equal to the size of `PhysicalExpr::children()`.
/// Allow the vtable address comparisons for PhysicalExpr Trait Objects，it is harmless even
/// in the case of 'false-native'.
#[allow(clippy::vtable_address_comparisons)]
pub fn with_new_children_if_necessary(
    expr: Arc<dyn PhysicalExpr>,
    children: Vec<Arc<dyn PhysicalExpr>>,
) -> Result<Arc<dyn PhysicalExpr>> {
    if children.len() != expr.children().len() {
        Err(DataFusionError::Internal(
            "PhysicalExpr: Wrong number of children".to_string(),
        ))
    } else if children.is_empty()
        || children
            .iter()
            .zip(expr.children().iter())
            .any(|(c1, c2)| !Arc::ptr_eq(c1, c2))
    {
        expr.with_new_children(children)
    } else {
        Ok(expr)
    }
}

pub fn down_cast_any_ref(any: &dyn Any) -> &dyn Any {
    if any.is::<Arc<dyn PhysicalExpr>>() {
        any.downcast_ref::<Arc<dyn PhysicalExpr>>()
            .unwrap()
            .as_any()
    } else if any.is::<Box<dyn PhysicalExpr>>() {
        any.downcast_ref::<Box<dyn PhysicalExpr>>()
            .unwrap()
            .as_any()
    } else {
        any
    }
}

/// Scatter `truthy` array by boolean mask. When the mask evaluates `true`, next values of `truthy`
/// are taken, when the mask evaluates `false` values null values are filled.
///
/// # Arguments
/// * `mask` - Boolean values used to determine where to put the `truthy` values
/// * `truthy` - All values of this array are to scatter according to `mask` into final result.
fn scatter(mask: &BooleanArray, truthy: &dyn Array) -> Result<ArrayRef> {
    let truthy = truthy.data();

    // update the mask so that any null values become false
    // (SlicesIterator doesn't respect nulls)
    let mask = and_kleene(mask, &is_not_null(mask)?)?;

    let mut mutable = MutableArrayData::new(vec![truthy], true, mask.len());

    // the SlicesIterator slices only the true values. So the gaps left by this iterator we need to
    // fill with falsy values

    // keep track of how much is filled
    let mut filled = 0;
    // keep track of current position we have in truthy array
    let mut true_pos = 0;

    SlicesIterator::new(&mask).for_each(|(start, end)| {
        // the gap needs to be filled with nulls
        if start > filled {
            mutable.extend_nulls(start - filled);
        }
        // fill with truthy values
        let len = end - start;
        mutable.extend(0, true_pos, true_pos + len);
        true_pos += len;
        filled = end;
    });
    // the remaining part is falsy
    if filled < mask.len() {
        mutable.extend_nulls(mask.len() - filled);
    }

    let data = mutable.freeze();
    Ok(make_array(data))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow::array::Int32Array;
    use datafusion_common::Result;

    #[test]
    fn scatter_int() -> Result<()> {
        let truthy = Arc::new(Int32Array::from(vec![1, 10, 11, 100]));
        let mask = BooleanArray::from(vec![true, true, false, false, true]);

        // the output array is expected to be the same length as the mask array
        let expected =
            Int32Array::from_iter(vec![Some(1), Some(10), None, None, Some(11)]);
        let result = scatter(&mask, truthy.as_ref())?;
        let result = result.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(&expected, result);
        Ok(())
    }

    #[test]
    fn scatter_int_end_with_false() -> Result<()> {
        let truthy = Arc::new(Int32Array::from(vec![1, 10, 11, 100]));
        let mask = BooleanArray::from(vec![true, false, true, false, false, false]);

        // output should be same length as mask
        let expected =
            Int32Array::from_iter(vec![Some(1), None, Some(10), None, None, None]);
        let result = scatter(&mask, truthy.as_ref())?;
        let result = result.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(&expected, result);
        Ok(())
    }

    #[test]
    fn scatter_with_null_mask() -> Result<()> {
        let truthy = Arc::new(Int32Array::from(vec![1, 10, 11]));
        let mask: BooleanArray = vec![Some(false), None, Some(true), Some(true), None]
            .into_iter()
            .collect();

        // output should treat nulls as though they are false
        let expected = Int32Array::from_iter(vec![None, None, Some(1), Some(10), None]);
        let result = scatter(&mask, truthy.as_ref())?;
        let result = result.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(&expected, result);
        Ok(())
    }

    #[test]
    fn scatter_boolean() -> Result<()> {
        let truthy = Arc::new(BooleanArray::from(vec![false, false, false, true]));
        let mask = BooleanArray::from(vec![true, true, false, false, true]);

        // the output array is expected to be the same length as the mask array
        let expected = BooleanArray::from_iter(vec![
            Some(false),
            Some(false),
            None,
            None,
            Some(false),
        ]);
        let result = scatter(&mask, truthy.as_ref())?;
        let result = result.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(&expected, result);
        Ok(())
    }

    #[test]
    fn reduce_boundaries() -> Result<()> {
        let different_boundaries = ExprBoundaries::new(
            ScalarValue::Int32(Some(1)),
            ScalarValue::Int32(Some(10)),
            None,
        );
        assert_eq!(different_boundaries.reduce(), None);

        let scalar_boundaries = ExprBoundaries::new(
            ScalarValue::Int32(Some(1)),
            ScalarValue::Int32(Some(1)),
            None,
        );
        assert_eq!(
            scalar_boundaries.reduce(),
            Some(ScalarValue::Int32(Some(1)))
        );

        // Can still reduce.
        let no_boundaries =
            ExprBoundaries::new(ScalarValue::Int32(None), ScalarValue::Int32(None), None);
        assert_eq!(no_boundaries.reduce(), Some(ScalarValue::Int32(None)));

        Ok(())
    }
}
