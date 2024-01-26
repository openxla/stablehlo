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

// This is the canonicalize pattern definition file.

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/StablehloOps.td"
include "stablehlo/dialect/StablehloUtils.td"


// A dynamic reshape of a dynamic reshape is a dynamic reshape.
def RemoveRedundantDynamicReshape : Pat<
  (StableHLO_DynamicReshapeOp (StableHLO_DynamicReshapeOp $operand, $shape1), $shape2),
  (StableHLO_DynamicReshapeOp $operand, $shape2)>;

// A dynamic broadcast of a dynamic reshape with the same shape operand
// is a dynamic reshape.
def RemoveRedundantDynamicBroadcast : Pat<
  (StableHLO_DynamicBroadcastInDimOp
    (StableHLO_DynamicReshapeOp $operand, $shape),
    $shape,
    IdentityBroadcastDims,
    $known_expanding_dimensions,
    $known_nonexpanding_dimensions),
  (StableHLO_DynamicReshapeOp $operand, $shape)>;

