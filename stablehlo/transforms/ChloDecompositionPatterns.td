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
def : Pat<(CHLO_AtanOp NonComplexElementType:$input),
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


def : Pat<(CHLO_TanOp $input),
          (StableHLO_TanOp $input)>;

def : Pat<(CHLO_ConstantOp $v),
          (StableHLO_ConstantOp $v)>;


// Include generated implementations of math functions:
include "ChloDecompositionPatternsMath.td"
