// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This is the legalization pattern definition file for CHLO to StableHLO.
// These are included in the populateDecompositionPatterns factory
// and should only include canonical expansions which are not actually
// ambiguous/different for various backends. Avoid patterns that are actually
// lowering to non-canonical forms.

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/ChloOps.td"
include "stablehlo/dialect/StablehloOps.td"

class StableHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<StableHLO_ComparisonDirectionAttr,
               "::mlir::stablehlo::ComparisonDirection::" # enumStr>;

class StableHLO_ConstantLike<string value> : NativeCodeCall<
    "::mlir::stablehlo::getConstantLike($_builder, $_loc, " # value # ", $0)">;

def ComplexElementType : Type<
  CPred<"isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Complex element type">;

def NonComplexElementType : Type<
  CPred<"!isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Non-complex element type">;

def StableHLO_ConstantLikeMaxFiniteValue : NativeCodeCall<
    "::mlir::stablehlo::getConstantLikeMaxFiniteValue($_builder, $_loc, $0)">;

def StableHLO_ConstantLikePosInfValue : NativeCodeCall<
    "::mlir::stablehlo::getConstantLikeInfValue($_builder, $_loc, $0, /*negative=*/false)">;

def StableHLO_ConstantLikeNegInfValue : NativeCodeCall<
    "::mlir::stablehlo::getConstantLikeInfValue($_builder, $_loc, $0, /*negative=*/true)">;

def StableHLO_ConstantLikeSmallestNormalizedValue : NativeCodeCall<
    "::mlir::stablehlo::getConstantLikeSmallestNormalizedValue($_builder, $_loc, $0)">;


//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

// Express `atan` as
//   atan(x) = atan2(x, 1)
def : Pat<(CHLO_AtanOp $input),
  (StableHLO_Atan2Op
    $input,
    (StableHLO_ConstantLike<"1"> $input)
  )>;

// Express `atanh` for non-complex arguments as follows:
//   atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) <= 1
//   atanh(x) = nan                          otherwise
def : Pat<(CHLO_AtanhOp NonComplexElementType:$input),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
      (StableHLO_AbsOp $input),
      (StableHLO_ConstantLike<"1"> $input),
      StableHLO_ComparisonDirectionValue<"GT">,
      (STABLEHLO_DEFAULT_COMPARISON_TYPE)
    ),
    (StableHLO_ConstantLike<"NAN"> $input),
    (StableHLO_MulOp
      (StableHLO_SubtractOp
        (StableHLO_Log1pOp $input),
        (StableHLO_Log1pOp
          (StableHLO_NegOp $input)
        )
      ),
      (StableHLO_ConstantLike<"0.5"> $input)
    )
  )>;

// Express `atanh` for complex arguments as follows:
//   atanh(x) = (log(1 + x) - log(1 + (-x))) * 0.5
//
// Per tensorflow/compiler/xla/client/lib/math.cc at the time of writing:
// "For now, we ignore the nan edge case for complex inputs,
// because we don't yet have exhaustive tests for complex trig functions".
def : Pat<(CHLO_AtanhOp ComplexElementType:$input),
  (StableHLO_MulOp
    (StableHLO_SubtractOp
      (StableHLO_Log1pOp $input),
      (StableHLO_Log1pOp
        (StableHLO_NegOp $input)
      )
    ),
    (StableHLO_ConstantLike<"0.5"> $input)
  )>;

// Express `conj` as
//   conj(x) = (re(x), -im(x)).
def : Pat<(CHLO_ConjOp $v),
          (StableHLO_ComplexOp (StableHLO_RealOp $v), (StableHLO_NegOp (StableHLO_ImagOp $v)))>;

// Express `is_inf` as
//   is_inf(x) = is_pos_inf(|x|)
def : Pat<(CHLO_IsInfOp NonComplexElementType:$input),
  (CHLO_IsPosInfOp
    (StableHLO_AbsOp $input)
  )>;

// Express `is_pos_inf` as
//   is_pos_inf(x) = (x == +inf)
def : Pat<(CHLO_IsPosInfOp NonComplexElementType:$input),
  (StableHLO_CompareOp
    $input,
    (StableHLO_ConstantLikePosInfValue $input),
    StableHLO_ComparisonDirectionValue<"EQ">,
    (STABLEHLO_DEFAULT_COMPARISON_TYPE)
  )>;

// Express `is_neg_inf` as
//   is_neg_inf(x) = (x == -inf)
def : Pat<(CHLO_IsNegInfOp NonComplexElementType:$input),
  (StableHLO_CompareOp
    $input,
    (StableHLO_ConstantLikeNegInfValue $input),
    StableHLO_ComparisonDirectionValue<"EQ">,
    (STABLEHLO_DEFAULT_COMPARISON_TYPE)
  )>;

// Express `tan` as
//   sine(x) / cosine(x)
def : Pat<(CHLO_TanOp NonComplexElementType:$input),
  (StableHLO_DivOp
    (StableHLO_SineOp $input),
    (StableHLO_CosineOp $input)
  )>;


// Express `tan(a + bi)` as
//   (tan(a) + i tanh(b)) / (1 - i tan(a) * tanh(b))
def : Pat<(CHLO_TanOp ComplexElementType:$input),
  (StableHLO_DivOp
    (StableHLO_ComplexOp
      (CHLO_TanOp:$tan (StableHLO_RealOp $input)),
      (StableHLO_TanhOp:$tanh (StableHLO_ImagOp $input))),
    (StableHLO_ComplexOp
      (StableHLO_ConstantLike<"1.0"> $tan),
      (StableHLO_NegOp (StableHLO_MulOp $tan, $tanh)))
  )>;

def : Pat<(CHLO_ConstantOp $v),
          (StableHLO_ConstantOp $v)>;


// Include generated implementations of math functions:
include "ChloDecompositionPatternsMath.td"
