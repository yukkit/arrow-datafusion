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

//! Simplify expressions optimizer rule and implementation

use super::{ExprSimplifier, SimplifyContext};
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_common::Result;
use datafusion_expr::{logical_plan::LogicalPlan, utils::from_plan};
use datafusion_physical_expr::execution_props::ExecutionProps;

/// Optimizer Pass that simplifies [`LogicalPlan`]s by rewriting
/// [`Expr`]`s evaluating constants and applying algebraic
/// simplifications
///
/// # Introduction
/// It uses boolean algebra laws to simplify or reduce the number of terms in expressions.
///
/// # Example:
/// `Filter: b > 2 AND b > 2`
/// is optimized to
/// `Filter: b > 2`
///
#[derive(Default)]
pub struct SimplifyExpressions {}

impl OptimizerRule for SimplifyExpressions {
    fn name(&self) -> &str {
        "simplify_expressions"
    }

    fn optimize(
        &self,
        plan: &LogicalPlan,
        optimizer_config: &mut OptimizerConfig,
    ) -> Result<LogicalPlan> {
        let mut execution_props = ExecutionProps::new();
        execution_props.query_execution_start_time =
            optimizer_config.query_execution_start_time();
        self.optimize_internal(plan, &execution_props)
    }
}

impl SimplifyExpressions {
    fn optimize_internal(
        &self,
        plan: &LogicalPlan,
        execution_props: &ExecutionProps,
    ) -> Result<LogicalPlan> {
        // We need to pass down the all schemas within the plan tree to `optimize_expr` in order to
        // to evaluate expression types. For example, a projection plan's schema will only include
        // projected columns. With just the projected schema, it's not possible to infer types for
        // expressions that references non-projected columns within the same project plan or its
        // children plans.
        let info = plan
            .all_schemas()
            .into_iter()
            .fold(SimplifyContext::new(execution_props), |context, schema| {
                context.with_schema(schema.clone())
            });

        let simplifier = ExprSimplifier::new(info);

        let new_inputs = plan
            .inputs()
            .iter()
            .map(|input| self.optimize_internal(input, execution_props))
            .collect::<Result<Vec<_>>>()?;

        let expr = plan
            .expressions()
            .into_iter()
            .map(|e| {
                // We need to keep original expression name, if any.
                // Constant folding should not change expression name.
                let name = &e.display_name();

                // Apply the actual simplification logic
                let new_e = simplifier.simplify(e)?;

                let new_name = &new_e.display_name();

                if let (Ok(expr_name), Ok(new_expr_name)) = (name, new_name) {
                    if expr_name != new_expr_name {
                        Ok(new_e.alias(expr_name))
                    } else {
                        Ok(new_e)
                    }
                } else {
                    Ok(new_e)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        from_plan(plan, &expr, &new_inputs)
    }
}

impl SimplifyExpressions {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(test)]
mod tests {
    use crate::simplify_expressions::utils::for_test::{
        cast_to_int64_expr, now_expr, to_timestamp_expr,
    };

    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use chrono::{DateTime, TimeZone, Utc};
    use datafusion_common::ScalarValue;
    use datafusion_expr::{or, Between, BinaryExpr, Cast, Operator};

    use datafusion_expr::logical_plan::table_scan;
    use datafusion_expr::{
        and, binary_expr, col, lit, logical_plan::builder::LogicalPlanBuilder, Expr,
        ExprSchemable,
    };

    /// A macro to assert that one string is contained within another with
    /// a nice error message if they are not.
    ///
    /// Usage: `assert_contains!(actual, expected)`
    ///
    /// Is a macro so test error
    /// messages are on the same line as the failure;
    ///
    /// Both arguments must be convertable into Strings (Into<String>)
    macro_rules! assert_contains {
        ($ACTUAL: expr, $EXPECTED: expr) => {
            let actual_value: String = $ACTUAL.into();
            let expected_value: String = $EXPECTED.into();
            assert!(
                actual_value.contains(&expected_value),
                "Can not find expected in actual.\n\nExpected:\n{}\n\nActual:\n{}",
                expected_value,
                actual_value
            );
        };
    }

    fn test_table_scan() -> LogicalPlan {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Boolean, false),
            Field::new("b", DataType::Boolean, false),
            Field::new("c", DataType::Boolean, false),
            Field::new("d", DataType::UInt32, false),
        ]);
        table_scan(Some("test"), &schema, None)
            .expect("creating scan")
            .build()
            .expect("building plan")
    }

    fn assert_optimized_plan_eq(plan: &LogicalPlan, expected: &str) {
        let rule = SimplifyExpressions::new();
        let optimized_plan = rule
            .optimize(plan, &mut OptimizerConfig::new())
            .expect("failed to optimize plan");
        let formatted_plan = format!("{:?}", optimized_plan);
        assert_eq!(formatted_plan, expected);
    }

    #[test]
    fn test_simplify_optimized_plan() {
        let table_scan = test_table_scan();
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a")])
            .unwrap()
            .filter(and(col("b").gt(lit(1)), col("b").gt(lit(1))))
            .unwrap()
            .build()
            .unwrap();

        assert_optimized_plan_eq(
            &plan,
            "\
	        Filter: test.b > Int32(1)\
            \n  Projection: test.a\
            \n    TableScan: test",
        );
    }

    #[test]
    fn test_simplify_optimized_plan_with_or() {
        let table_scan = test_table_scan();
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a")])
            .unwrap()
            .filter(or(col("b").gt(lit(1)), col("b").gt(lit(1))))
            .unwrap()
            .build()
            .unwrap();

        assert_optimized_plan_eq(
            &plan,
            "\
            Filter: test.b > Int32(1)\
            \n  Projection: test.a\
            \n    TableScan: test",
        );
    }

    #[test]
    fn test_simplify_optimized_plan_with_composed_and() {
        let table_scan = test_table_scan();
        // ((c > 5) AND (d < 6)) AND (c > 5) --> (c > 5) AND (d < 6)
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])
            .unwrap()
            .filter(and(
                and(col("a").gt(lit(5)), col("b").lt(lit(6))),
                col("a").gt(lit(5)),
            ))
            .unwrap()
            .build()
            .unwrap();

        assert_optimized_plan_eq(
            &plan,
            "\
            Filter: test.a > Int32(5) AND test.b < Int32(6)\
            \n  Projection: test.a, test.b\
	        \n    TableScan: test",
        );
    }

    #[test]
    fn test_simplity_optimized_plan_eq_expr() {
        let table_scan = test_table_scan();
        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("b").eq(lit(true)))
            .unwrap()
            .filter(col("c").eq(lit(false)))
            .unwrap()
            .project(vec![col("a")])
            .unwrap()
            .build()
            .unwrap();

        let expected = "\
        Projection: test.a\
        \n  Filter: NOT test.c\
        \n    Filter: test.b\
        \n      TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn test_simplity_optimized_plan_not_eq_expr() {
        let table_scan = test_table_scan();
        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("b").not_eq(lit(true)))
            .unwrap()
            .filter(col("c").not_eq(lit(false)))
            .unwrap()
            .limit(0, Some(1))
            .unwrap()
            .project(vec![col("a")])
            .unwrap()
            .build()
            .unwrap();

        let expected = "\
        Projection: test.a\
        \n  Limit: skip=0, fetch=1\
        \n    Filter: test.c\
        \n      Filter: NOT test.b\
        \n        TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn test_simplity_optimized_plan_and_expr() {
        let table_scan = test_table_scan();
        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("b").not_eq(lit(true)).and(col("c").eq(lit(true))))
            .unwrap()
            .project(vec![col("a")])
            .unwrap()
            .build()
            .unwrap();

        let expected = "\
        Projection: test.a\
        \n  Filter: NOT test.b AND test.c\
        \n    TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn test_simplity_optimized_plan_or_expr() {
        let table_scan = test_table_scan();
        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("b").not_eq(lit(true)).or(col("c").eq(lit(false))))
            .unwrap()
            .project(vec![col("a")])
            .unwrap()
            .build()
            .unwrap();

        let expected = "\
        Projection: test.a\
        \n  Filter: NOT test.b OR NOT test.c\
        \n    TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn test_simplity_optimized_plan_not_expr() {
        let table_scan = test_table_scan();
        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("b").eq(lit(false)).not())
            .unwrap()
            .project(vec![col("a")])
            .unwrap()
            .build()
            .unwrap();

        let expected = "\
        Projection: test.a\
        \n  Filter: test.b\
        \n    TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn test_simplity_optimized_plan_support_projection() {
        let table_scan = test_table_scan();
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("d"), col("b").eq(lit(false))])
            .unwrap()
            .build()
            .unwrap();

        let expected = "\
        Projection: test.a, test.d, NOT test.b AS test.b = Boolean(false)\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn test_simplity_optimized_plan_support_aggregate() {
        let table_scan = test_table_scan();
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("c"), col("b")])
            .unwrap()
            .aggregate(
                vec![col("a"), col("c")],
                vec![
                    datafusion_expr::max(col("b").eq(lit(true))),
                    datafusion_expr::min(col("b")),
                ],
            )
            .unwrap()
            .build()
            .unwrap();

        let expected = "\
        Aggregate: groupBy=[[test.a, test.c]], aggr=[[MAX(test.b) AS MAX(test.b = Boolean(true)), MIN(test.b)]]\
        \n  Projection: test.a, test.c, test.b\
        \n    TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn test_simplity_optimized_plan_support_values() {
        let expr1 = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(lit(1)),
            Operator::Plus,
            Box::new(lit(2)),
        ));
        let expr2 = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(lit(2)),
            Operator::Minus,
            Box::new(lit(1)),
        ));
        let values = vec![vec![expr1, expr2]];
        let plan = LogicalPlanBuilder::values(values).unwrap().build().unwrap();

        let expected = "\
        Values: (Int32(3) AS Int32(1) + Int32(2), Int32(1) AS Int32(2) - Int32(1))";

        assert_optimized_plan_eq(&plan, expected);
    }

    // expect optimizing will result in an error, returning the error string
    fn get_optimized_plan_err(plan: &LogicalPlan, date_time: &DateTime<Utc>) -> String {
        let mut config =
            OptimizerConfig::new().with_query_execution_start_time(*date_time);
        let rule = SimplifyExpressions::new();

        let err = rule
            .optimize(plan, &mut config)
            .expect_err("expected optimization to fail");

        err.to_string()
    }

    fn get_optimized_plan_formatted(
        plan: &LogicalPlan,
        date_time: &DateTime<Utc>,
    ) -> String {
        let mut config =
            OptimizerConfig::new().with_query_execution_start_time(*date_time);
        let rule = SimplifyExpressions::new();

        let optimized_plan = rule
            .optimize(plan, &mut config)
            .expect("failed to optimize plan");
        format!("{:?}", optimized_plan)
    }

    #[test]
    fn to_timestamp_expr_folded() {
        let table_scan = test_table_scan();
        let proj = vec![to_timestamp_expr("2020-09-08T12:00:00+00:00")];

        let plan = LogicalPlanBuilder::from(table_scan)
            .project(proj)
            .unwrap()
            .build()
            .unwrap();

        let expected = "Projection: TimestampNanosecond(1599566400000000000, None) AS totimestamp(Utf8(\"2020-09-08T12:00:00+00:00\"))\
            \n  TableScan: test"
            .to_string();
        let actual = get_optimized_plan_formatted(&plan, &Utc::now());
        assert_eq!(expected, actual);
    }

    #[test]
    fn to_timestamp_expr_wrong_arg() {
        let table_scan = test_table_scan();
        let proj = vec![to_timestamp_expr("I'M NOT A TIMESTAMP")];
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(proj)
            .unwrap()
            .build()
            .unwrap();

        let expected = "Error parsing 'I'M NOT A TIMESTAMP' as timestamp";
        let actual = get_optimized_plan_err(&plan, &Utc::now());
        assert_contains!(actual, expected);
    }

    #[test]
    fn cast_expr() {
        let table_scan = test_table_scan();
        let proj = vec![Expr::Cast(Cast::new(Box::new(lit("0")), DataType::Int32))];
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(proj)
            .unwrap()
            .build()
            .unwrap();

        let expected = "Projection: Int32(0) AS Utf8(\"0\")\
            \n  TableScan: test";
        let actual = get_optimized_plan_formatted(&plan, &Utc::now());
        assert_eq!(expected, actual);
    }

    #[test]
    fn cast_expr_wrong_arg() {
        let table_scan = test_table_scan();
        let proj = vec![Expr::Cast(Cast::new(Box::new(lit("")), DataType::Int32))];
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(proj)
            .unwrap()
            .build()
            .unwrap();

        let expected = "Cannot cast string '' to value of Int32 type";
        let actual = get_optimized_plan_err(&plan, &Utc::now());
        assert_contains!(actual, expected);
    }

    #[test]
    fn multiple_now_expr() {
        let table_scan = test_table_scan();
        let time = Utc::now();
        let proj = vec![
            now_expr(),
            Expr::Alias(Box::new(now_expr()), "t2".to_string()),
        ];
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(proj)
            .unwrap()
            .build()
            .unwrap();

        // expect the same timestamp appears in both exprs
        let actual = get_optimized_plan_formatted(&plan, &time);
        let expected = format!(
            "Projection: TimestampNanosecond({}, Some(\"UTC\")) AS now(), TimestampNanosecond({}, Some(\"UTC\")) AS t2\
            \n  TableScan: test",
            time.timestamp_nanos(),
            time.timestamp_nanos()
        );

        assert_eq!(expected, actual);
    }

    #[test]
    fn simplify_and_eval() {
        // demonstrate a case where the evaluation needs to run prior
        // to the simplifier for it to work
        let table_scan = test_table_scan();
        let time = Utc::now();
        // (true or false) != col --> !col
        let proj = vec![lit(true).or(lit(false)).not_eq(col("a"))];
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(proj)
            .unwrap()
            .build()
            .unwrap();

        let actual = get_optimized_plan_formatted(&plan, &time);
        let expected =
            "Projection: NOT test.a AS Boolean(true) OR Boolean(false) != test.a\
                        \n  TableScan: test";

        assert_eq!(expected, actual);
    }

    #[test]
    fn now_less_than_timestamp() {
        let table_scan = test_table_scan();

        let ts_string = "2020-09-08T12:05:00+00:00";
        let time = chrono::Utc.timestamp_nanos(1599566400000000000i64);

        //  cast(now() as int) < cast(to_timestamp(...) as int) + 50000_i64
        let plan =
            LogicalPlanBuilder::from(table_scan)
                .filter(
                    cast_to_int64_expr(now_expr())
                        .lt(cast_to_int64_expr(to_timestamp_expr(ts_string))
                            + lit(50000_i64)),
                )
                .unwrap()
                .build()
                .unwrap();

        // Note that constant folder runs and folds the entire
        // expression down to a single constant (true)
        let expected = "Filter: Boolean(true)\
                        \n  TableScan: test";
        let actual = get_optimized_plan_formatted(&plan, &time);

        assert_eq!(expected, actual);
    }

    #[test]
    fn select_date_plus_interval() {
        let table_scan = test_table_scan();

        let ts_string = "2020-09-08T12:05:00+00:00";
        let time = chrono::Utc.timestamp_nanos(1599566400000000000i64);

        //  now() < cast(to_timestamp(...) as int) + 5000000000
        let schema = table_scan.schema();

        let date_plus_interval_expr = to_timestamp_expr(ts_string)
            .cast_to(&DataType::Date32, schema)
            .unwrap()
            + Expr::Literal(ScalarValue::IntervalDayTime(Some(123i64 << 32)));

        let plan = LogicalPlanBuilder::from(table_scan.clone())
            .project(vec![date_plus_interval_expr])
            .unwrap()
            .build()
            .unwrap();

        println!("{:?}", plan);

        // Note that constant folder runs and folds the entire
        // expression down to a single constant (true)
        let expected = r#"Projection: Date32("18636") AS totimestamp(Utf8("2020-09-08T12:05:00+00:00")) + IntervalDayTime("528280977408")
  TableScan: test"#;
        let actual = get_optimized_plan_formatted(&plan, &time);

        assert_eq!(expected, actual);
    }

    #[test]
    fn simplify_not_binary() {
        let table_scan = test_table_scan();

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("d").gt(lit(10)).not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d <= Int32(10)\
            \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_bool_and() {
        let table_scan = test_table_scan();

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("d").gt(lit(10)).and(col("d").lt(lit(100))).not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d <= Int32(10) OR test.d >= Int32(100)\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_bool_or() {
        let table_scan = test_table_scan();

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("d").gt(lit(10)).or(col("d").lt(lit(100))).not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d <= Int32(10) AND test.d >= Int32(100)\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_not() {
        let table_scan = test_table_scan();

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("d").gt(lit(10)).not().not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d > Int32(10)\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_null() {
        let table_scan = test_table_scan();

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("d").is_null().not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d IS NOT NULL\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_not_null() {
        let table_scan = test_table_scan();

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("d").is_not_null().not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d IS NULL\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_in() {
        let table_scan = test_table_scan();

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("d").in_list(vec![lit(1), lit(2), lit(3)], false).not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d NOT IN ([Int32(1), Int32(2), Int32(3)])\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_not_in() {
        let table_scan = test_table_scan();

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("d").in_list(vec![lit(1), lit(2), lit(3)], true).not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d IN ([Int32(1), Int32(2), Int32(3)])\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_between() {
        let table_scan = test_table_scan();
        let qual = Expr::Between(Between::new(
            Box::new(col("d")),
            false,
            Box::new(lit(1)),
            Box::new(lit(10)),
        ));

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(qual.not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d < Int32(1) OR test.d > Int32(10)\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_not_between() {
        let table_scan = test_table_scan();
        let qual = Expr::Between(Between::new(
            Box::new(col("d")),
            true,
            Box::new(lit(1)),
            Box::new(lit(10)),
        ));

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(qual.not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d >= Int32(1) AND test.d <= Int32(10)\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_like() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
        ]);
        let table_scan = table_scan(Some("test"), &schema, None)
            .expect("creating scan")
            .build()
            .expect("building plan");

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("a").like(col("b")).not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.a NOT LIKE test.b\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_not_like() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
        ]);
        let table_scan = table_scan(Some("test"), &schema, None)
            .expect("creating scan")
            .build()
            .expect("building plan");

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(col("a").not_like(col("b")).not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.a LIKE test.b\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_distinct_from() {
        let table_scan = test_table_scan();

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(binary_expr(col("d"), Operator::IsDistinctFrom, lit(10)).not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d IS NOT DISTINCT FROM Int32(10)\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }

    #[test]
    fn simplify_not_not_distinct_from() {
        let table_scan = test_table_scan();

        let plan = LogicalPlanBuilder::from(table_scan)
            .filter(binary_expr(col("d"), Operator::IsNotDistinctFrom, lit(10)).not())
            .unwrap()
            .build()
            .unwrap();
        let expected = "Filter: test.d IS DISTINCT FROM Int32(10)\
        \n  TableScan: test";

        assert_optimized_plan_eq(&plan, expected);
    }
}
