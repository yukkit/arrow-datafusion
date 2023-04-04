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

//! NamedStruct expressions for physical operations

use std::any::Any;
use std::sync::Arc;

use arrow::{
    array::StructArray,
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};
use arrow_schema::Field;

use crate::physical_expr::down_cast_any_ref;
use crate::PhysicalExpr;
use datafusion_common::Result;
use datafusion_expr::ColumnarValue;

/// Represents a literal value
#[derive(Debug)]
pub struct NamedStruct {
    // exprs: Vec<(Field, Arc<dyn PhysicalExpr>)>,
    exprs: Vec<(String, Arc<dyn PhysicalExpr>)>,
}

impl NamedStruct {
    /// Create a literal value expression
    pub fn new(exprs: Vec<(String, Arc<dyn PhysicalExpr>)>) -> Self {
        Self { exprs }
    }
}

impl std::fmt::Display for NamedStruct {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "NamedStruct ({:?})", self.exprs)
    }
}

impl PhysicalExpr for NamedStruct {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        let fields = self
            .exprs
            .iter()
            .map(|(name, expr)| {
                let data_type = expr.data_type(input_schema)?;
                let nullable = expr.nullable(input_schema)?;
                Ok(Field::new(name, data_type, nullable))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DataType::Struct(fields))
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let data = self
            .exprs
            .iter()
            .map(|(name, expr)| Ok((name.as_str(), expr.evaluate(batch)?.into_array(1))))
            .collect::<Result<Vec<_>>>()?;

        Ok(ColumnarValue::Array(Arc::new(StructArray::try_from(data)?)))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.exprs
            .iter()
            .map(|(_, expr)| expr.clone())
            .collect::<Vec<_>>()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let exprs = children
            .into_iter()
            .zip(self.exprs.iter())
            .map(|(child, (name, _))| (name.clone(), child))
            .collect::<Vec<_>>();

        Ok(Arc::new(Self::new(exprs)))
    }
}

impl PartialEq<dyn Any> for NamedStruct {
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| {
                self.exprs.len() == x.exprs.len()
                    && self
                        .exprs
                        .iter()
                        .zip(x.exprs.iter())
                        .all(|(a, b)| a.0 == b.0 && a.1.eq(&b.1))
            })
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use crate::{expressions::Column, PhysicalExpr};

    use super::NamedStruct;

    #[test]
    fn test() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
        let a = Int32Array::from(vec![Some(1), None, Some(3), Some(4), Some(5)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a)]).unwrap();

        let expr = NamedStruct::new(vec![
            ("a".to_string(), Arc::new(Column::new("a", 0))),
            ("b".to_string(), Arc::new(Column::new("a", 0))),
        ]);

        let batch = expr.evaluate(&batch).unwrap();

        println!("{batch:?}")
    }
}
