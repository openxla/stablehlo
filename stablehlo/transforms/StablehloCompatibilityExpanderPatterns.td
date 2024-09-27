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

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/StablehloOps.td"

def ComplexElementType : Type<
  CPred<"isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Complex element type">;

def NonComplexElementType : Type<
  CPred<"!isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Non-complex element type">;

class StableHLO_ConstantLike<string value> : NativeCodeCall<
    "::mlir::stablehlo::getConstantLike($_builder, $_loc, " # value # ", $0)">;

// Express `tan` as
//   sine(x) / cosine(x)
def TanOp_CompatiblityExpander : Pat<(StableHLO_TanOp NonComplexElementType:$input),
  (StableHLO_DivOp
    (StableHLO_SineOp $input),
    (StableHLO_CosineOp $input)
  )>;

// Express `tan(a + bi)` as
//   (tan(a) + i tanh(b)) / (1 - i tan(a) * tanh(b))
def TanOp_ComplexElementType_CompatiblityExpander : Pat<(StableHLO_TanOp ComplexElementType:$input),
  (StableHLO_DivOp
    (StableHLO_ComplexOp
      (StableHLO_TanOp:$tan (StableHLO_RealOp $input)),
      (StableHLO_TanhOp:$tanh (StableHLO_ImagOp $input))),
    (StableHLO_ComplexOp
      (StableHLO_ConstantLike<"1.0"> $tan),
      (StableHLO_NegOp (StableHLO_MulOp $tan, $tanh)))
  )>;
