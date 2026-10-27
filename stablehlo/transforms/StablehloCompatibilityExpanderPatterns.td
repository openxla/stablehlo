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

def FloatElementType : Type<
  CPred<"isa<FloatType>(cast<ShapedType>($_self).getElementType())">,
  "Float element type">;

def NonComplexElementType : Type<
  CPred<"!isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Non-complex element type">;

class StableHLO_ConstantLike<string value> : NativeCodeCall<
    "::mlir::stablehlo::getConstantLike($_builder, $_loc, " # value # ", $0)">;

def ConstDefaultResultAccuracyAttr :
  ConstantAttr<StableHLO_ResultAccuracyAttr, "::mlir::stablehlo::ResultAccuracyMode::DEFAULT">;

// Express `tan` as
//   sine(x) / cosine(x)
def TanOp_CompatiblityExpander : Pat<(StableHLO_TanOp NonComplexElementType:$input, ConstDefaultResultAccuracyAttr),
  (StableHLO_DivOp
    (StableHLO_SineOp $input, ConstDefaultResultAccuracyAttr),
    (StableHLO_CosineOp $input, ConstDefaultResultAccuracyAttr)
  )>;

// Express `tan(a + bi)` as
//   (tan(a) + i tanh(b)) / (1 - i tan(a) * tanh(b))
def TanOp_ComplexElementType_CompatiblityExpander : Pat<(StableHLO_TanOp ComplexElementType:$input, ConstDefaultResultAccuracyAttr),
  (StableHLO_DivOp
    (StableHLO_ComplexOp
      (StableHLO_TanOp:$tan (StableHLO_RealOp $input), ConstDefaultResultAccuracyAttr),
      (StableHLO_TanhOp:$tanh (StableHLO_ImagOp $input), ConstDefaultResultAccuracyAttr)),
    (StableHLO_ComplexOp
      (StableHLO_ConstantLike<"1.0"> $tan),
      (StableHLO_NegOp (StableHLO_MulOp $tan, $tanh)))
  )>;

// Express `exp2` as
//   exponential(x * ln(2))
def Exp2Op_CompatibilityExpander : Pat<
  (StableHLO_Exp2Op FloatElementType:$input, ConstDefaultResultAccuracyAttr),
  (StableHLO_ExpOp
    (StableHLO_MulOp
      $input,
      (StableHLO_ConstantLike<"llvm::numbers::ln2"> $input)),
    ConstDefaultResultAccuracyAttr)>;

// Express `exp2(a + bi)` as
//   exponential((a * ln(2)) + i (b * ln(2)))
def Exp2Op_ComplexElementType_CompatibilityExpander : Pat<
  (StableHLO_Exp2Op ComplexElementType:$input, ConstDefaultResultAccuracyAttr),
  (StableHLO_ExpOp
    (StableHLO_ComplexOp
      (StableHLO_MulOp
        (StableHLO_RealOp:$real $input),
        (StableHLO_ConstantLike<"llvm::numbers::ln2">:$ln2 $real)),
      (StableHLO_MulOp
        (StableHLO_ImagOp $input),
        $ln2)),
    ConstDefaultResultAccuracyAttr)>;

// Express `log2` as
//   log(x) / ln(2)
def Log2Op_CompatibilityExpander : Pat<
  (StableHLO_Log2Op FloatElementType:$input, ConstDefaultResultAccuracyAttr),
  (StableHLO_DivOp
    (StableHLO_LogOp $input, ConstDefaultResultAccuracyAttr),
    (StableHLO_ConstantLike<"llvm::numbers::ln2"> $input))>;

// Express `log2(a + bi)` as
//   (real(log(a + bi)) / ln(2)) + i (imag(log(a + bi)) / ln(2))
def Log2Op_ComplexElementType_CompatibilityExpander : Pat<
  (StableHLO_Log2Op ComplexElementType:$input, ConstDefaultResultAccuracyAttr),
  (StableHLO_ComplexOp
    (StableHLO_DivOp
      (StableHLO_RealOp:$real (StableHLO_LogOp:$log $input, ConstDefaultResultAccuracyAttr)),
      (StableHLO_ConstantLike<"llvm::numbers::ln2">:$ln2 $real)),
    (StableHLO_DivOp
      (StableHLO_ImagOp $log),
      $ln2))>;
