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

def IsSignedIntegerType : Constraint<CPred<
  "!llvm::cast<IntegerType>(llvm::cast<ShapedType>($0.getType()).getElementType()).isUnsigned()">>;

def IsSignedIntegerLessThan64Bit : Constraint<CPred<
  "isIntegerLessThanNBits($0, /*isSigned=*/true, 64)">>;

def IsSigned64BitIntegerType : Constraint<CPred<
  "isIntegerWithNBits($0, /*isSigned=*/true, 64)">>;

def IsUnsignedIntegerType : Constraint<CPred<
  "llvm::cast<IntegerType>(llvm::cast<ShapedType>($0.getType()).getElementType()).isUnsigned()">>;

def IsUnsignedIntegerLessThan64Bit : Constraint<CPred<
  "isIntegerLessThanNBits($0, /*isSigned=*/false, 64)">>;

def IsUnsigned64BitIntegerType : Constraint<CPred<
  "isIntegerWithNBits($0, /*isSigned=*/false, 64)">>;

def StableHLO_ConvertOpUpcast : NativeCodeCall<
    "::mlir::stablehlo::getConvertOpUpcast($_builder, $_loc, $0)">;

def StableHLO_ConvertToUnsigned : NativeCodeCall<
    "::mlir::stablehlo::getConvertToUnsigned($_builder, $_loc, $0)">;

def GetShiftAmountConstant : NativeCodeCall<
    "::mlir::stablehlo::getConstantLike($_builder, $_loc,"
    " cast<IntegerType>(cast<ShapedType>($0.getType()).getElementType()).getWidth(), $1)">;

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

def ConstDefaultResultAccuracyAttr :
  ConstantAttr<StableHLO_ResultAccuracyAttr, "::mlir::stablehlo::ResultAccuracyMode::DEFAULT">;

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
        (StableHLO_Log1pOp $input, ConstDefaultResultAccuracyAttr),
        (StableHLO_Log1pOp
          (StableHLO_NegOp $input), ConstDefaultResultAccuracyAttr
        )
      ),
      (StableHLO_ConstantLike<"0.5"> $input)
    )
  )>;

// Express `conj` as
//   conj(x) = (re(x), -im(x)).
def : Pat<(CHLO_ConjOp ComplexElementType:$v),
          (StableHLO_ComplexOp (StableHLO_RealOp $v), (StableHLO_NegOp (StableHLO_ImagOp $v)))>;

//   conj(x) = x if x is real
def : Pat<(CHLO_ConjOp NonComplexElementType:$v),
          (replaceWithValue $v)>;

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

// Signed mulhi_s32 for N-bit integers where N < 64, meaning we can upcast.
//   downcast((upcast(x, N*2) * upcast(y,N*2)) >> N, N)
def : Pat<(CHLO_MulhiOp $lhs, $rhs),
          (StableHLO_ConvertOp
            (StableHLO_ShiftRightArithmeticOp
              (StableHLO_MulOp:$wideMul (StableHLO_ConvertOpUpcast $lhs), (StableHLO_ConvertOpUpcast $rhs)),
              (GetShiftAmountConstant $lhs, $wideMul)
            )
          ),
          [(IsSignedIntegerLessThan64Bit $lhs)]>;

// Signed mulhi_s32 for N-bit integers where N < 64, meaning we can upcast.
//   downcast((upcast(x, N*2) * upcast(y,N*2)) >> N, N)
//
// Unsigned ints use logical shifts.
def : Pat<(CHLO_MulhiOp $lhs, $rhs),
          (StableHLO_ConvertOp
            (StableHLO_ShiftRightLogicalOp
              (StableHLO_MulOp:$wideMul (StableHLO_ConvertOpUpcast $lhs), (StableHLO_ConvertOpUpcast $rhs)),
              (GetShiftAmountConstant $lhs, $wideMul)
            )
          ),
          [(IsUnsignedIntegerLessThan64Bit $lhs)]>;

// mulhi_u64 with decomposition into 32 bit parts.
//
// 1. Split operands into lower and upper 32-bit halves
// uint64_t a_hi = a >> 32;
// uint64_t b_hi = b >> 32;
// uint64_t a_lo = a & 0xFFFFFFFF;
// uint64_t b_lo = b & 0xFFFFFFFF;

// // 2. Compute the four 64-bit partial products
// uint64_t p0 = a_lo * b_lo;
// uint64_t p1 = a_lo * b_hi;
// uint64_t p2 = a_hi * b_lo;
// uint64_t p3 = a_hi * b_hi;

// // 3. Accumulate middle terms and propagate the carry
// uint32_t carry_p0 = (uint32_t)(p0 >> 32);
// uint64_t mid1 = p1 + carry_p0;
// uint64_t mid2 = p2 + (uint32_t)mid1;

// // 4. Sum the high terms
// return p3 + (mid1 >> 32) + (mid2 >> 32);
def : Pat<(CHLO_MulhiOp $lhs, $rhs),
          (StableHLO_AddOp
            (StableHLO_AddOp
              (StableHLO_MulOp:$p3
                (StableHLO_ShiftRightLogicalOp:$a_hi $lhs, (StableHLO_ConstantLike<"32"> $lhs)),
                (StableHLO_ShiftRightLogicalOp:$b_hi $rhs, (StableHLO_ConstantLike<"32"> $rhs))
              ),
              (StableHLO_ShiftRightLogicalOp:$carry_mid1
                (StableHLO_AddOp:$mid1
                  (StableHLO_MulOp:$p1
                    (StableHLO_AndOp:$a_lo $lhs, (StableHLO_ConstantLike<"0xFFFFFFFFULL"> $lhs)),
                    $b_hi
                  ),
                  (StableHLO_ShiftRightLogicalOp:$carry_p0
                    (StableHLO_MulOp:$p0 $a_lo, (StableHLO_AndOp:$b_lo $rhs, (StableHLO_ConstantLike<"0xFFFFFFFFULL"> $rhs))),
                    (StableHLO_ConstantLike<"32"> $lhs)
                  )
                ),
                (StableHLO_ConstantLike<"32"> $lhs)
              )
            ),
            (StableHLO_ShiftRightLogicalOp:$carry_mid2
              (StableHLO_AddOp:$mid2
                (StableHLO_MulOp:$p2 $a_hi, $b_lo),
                (StableHLO_AndOp $mid1, (StableHLO_ConstantLike<"0xFFFFFFFFULL"> $lhs))
              ),
              (StableHLO_ConstantLike<"32"> $lhs)
            )
          ),
          [(IsUnsigned64BitIntegerType $lhs)]>;

// mulhi_u64 with correction for two's complement sign extension:
//   res = mulhi_u64((uint64_t)a, (uint64_t)b)
//   if (a < 0) hi -= b;
//   if (b < 0) hi -= a;
def : Pat<(CHLO_MulhiOp $lhs, $rhs),
          (StableHLO_ConvertOp
            (StableHLO_SubtractOp
              (StableHLO_SubtractOp
                (CHLO_MulhiOp:$hi
                  (StableHLO_ConvertToUnsigned $lhs),
                  (StableHLO_ConvertToUnsigned $rhs)
                ),
                (StableHLO_SelectOp
                  (StableHLO_CompareOp
                    $lhs,
                    (StableHLO_ConstantLike<"0"> $lhs),
                    StableHLO_ComparisonDirectionValue<"LT">,
                    (STABLEHLO_DEFAULT_COMPARISON_TYPE)
                  ),
                  (StableHLO_ConvertToUnsigned $rhs),
                  (StableHLO_ConstantLike<"0"> (StableHLO_ConvertToUnsigned $rhs))
                )
              ),
              (StableHLO_SelectOp
                (StableHLO_CompareOp
                  $rhs,
                  (StableHLO_ConstantLike<"0"> $rhs),
                  StableHLO_ComparisonDirectionValue<"LT">,
                  (STABLEHLO_DEFAULT_COMPARISON_TYPE)
                ),
                (StableHLO_ConvertToUnsigned $lhs),
                (StableHLO_ConstantLike<"0"> (StableHLO_ConvertToUnsigned $lhs))
              )
            )
          ),
          [(IsSigned64BitIntegerType $lhs)]>;

def : Pat<(CHLO_TanOp $input),
          (StableHLO_TanOp $input, ConstDefaultResultAccuracyAttr)>;

def : Pat<(CHLO_ConstantOp $v),
          (StableHLO_ConstantOp $v)>;


// Include generated implementations of math functions:
include "ChloDecompositionPatternsMath.td"
