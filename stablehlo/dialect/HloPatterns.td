/* Copyright 2022 The StableHLO Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Canonicalization patterns for the stablehlo dialect.

include "mlir/Dialect/Shape/IR/ShapeOps.td"
include "mlir/Dialect/Tensor/IR/TensorOps.td"
include "stablehlo/dialect/StablehloOps.td"

def HasSameType : Constraint<CPred<"$0.getType() == $1.getType()">>;

def ShapeOfDynamicReshape : Pat<
  (Shape_ShapeOfOp:$op (StableHLO_DynamicReshapeOp $x, $shape)),
  (replaceWithValue $shape),
  [(HasSameType $shape, $op)]>;
