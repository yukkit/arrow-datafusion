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

use crate::{Expr, LogicalPlan};
use arrow::datatypes::SchemaRef;
use std::any::Any;

///! Table source

/// Indicates whether and how a filter expression can be handled by a
/// TableProvider for table scans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableProviderFilterPushDown {
    /// The expression cannot be used by the provider.
    Unsupported,
    /// The expression can be used to help minimise the data retrieved,
    /// but the provider cannot guarantee that all returned tuples
    /// satisfy the filter. The Filter plan node containing this expression
    /// will be preserved.
    Inexact,
    /// The provider guarantees that all returned data satisfies this
    /// filter expression. The Filter plan node containing this expression
    /// will be removed.
    Exact,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableProviderAggregationPushDown {
    // Cannot push down
    Unsupported,
    // After pushing down the aggregate to the data source, the data source can still output data with
    // duplicated keys, which is OK as DataFusion will do GROUP BY key again.
    // The final query plan save `final aggregate` node.
    // Note that, if there is no grouping expression and the data source's partition is signal, need Ungrouped,
    Ungrouped,
    // After pushing down the aggregate to the data source, the data source can output data without
    // duplicated keys. The final query plan can remove `Aggregate` node.
    Grouped,
}

/// Indicates the type of this table for metadata/catalog purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableType {
    /// An ordinary physical table.
    Base,
    /// A non-materialised table that itself uses a query internally to provide data.
    View,
    /// A transient table.
    Temporary,
}

/// The TableSource trait is used during logical query planning and optimizations and
/// provides access to schema information and filter push-down capabilities. This trait
/// provides a subset of the functionality of the TableProvider trait in the core
/// datafusion crate. The TableProvider trait provides additional capabilities needed for
/// physical query execution (such as the ability to perform a scan). The reason for
/// having two separate traits is to avoid having the logical plan code be dependent
/// on the DataFusion execution engine. Other projects may want to use DataFusion's
/// logical plans and have their own execution engine.
pub trait TableSource: Sync + Send {
    fn as_any(&self) -> &dyn Any;

    /// Get a reference to the schema for this table
    fn schema(&self) -> SchemaRef;

    /// Get the type of this table for metadata/catalog purposes.
    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Tests whether the table provider can make use of a filter expression
    /// to optimise data retrieval.
    fn supports_filter_pushdown(
        &self,
        _filter: &Expr,
    ) -> datafusion_common::Result<TableProviderFilterPushDown> {
        Ok(TableProviderFilterPushDown::Unsupported)
    }

    /// true if the aggregation can be pushed down to datasource, false otherwise.
    fn supports_aggregate_pushdown(
        &self,
        _group_expr: &[Expr],
        _aggr_expr: &[Expr],
    ) -> datafusion_common::Result<TableProviderAggregationPushDown> {
        Ok(TableProviderAggregationPushDown::Unsupported)
    }

    /// Get the Logical plan of this table provider, if available.
    fn get_logical_plan(&self) -> Option<&LogicalPlan> {
        None
    }
}
