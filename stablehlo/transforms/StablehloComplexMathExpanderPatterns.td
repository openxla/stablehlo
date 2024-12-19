/* Copyright 2024 The StableHLO Authors.

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

//
// This file is generated using functional_algorithms tool (0.12.0).
// See build_tools/math/README.md for more information.

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/StablehloOps.td"

class StableHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<StableHLO_ComparisonDirectionAttr,
               "::mlir::stablehlo::ComparisonDirection::" # enumStr>;

class StableHLO_ConstantLike<string value> : NativeCodeCall<
    "::mlir::stablehlo::getConstantLike($_builder, $_loc, " # value # ", $0)">;

def ComplexElementType : Type<
  CPred<"isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Complex element type">;

def StableHLO_ConstantLikeMaxFiniteValue : NativeCodeCall<
    "::mlir::stablehlo::getConstantLikeMaxFiniteValue($_builder, $_loc, $0)">;

// Logarithm of 1 + z on complex input:
//
//       log1p(x + I * y) = 0.5 * log((x + 1) ** 2 + y ** 2) + I * arctan2(y, x + 1)
//
//     where
//
//       x and y are real and imaginary parts of the input to log1p, and
//       I is imaginary unit.
//
//     For evaluating the real part of log1p accurately on the whole
//     complex plane, the following cases must be handled separately:
//
//     A) Avoid catastrophic cancellation errors when x is close `-0.5 * y * y`
//        and `abs(y) < 1`.
//     B) Avoid overflow from square when x or y are large in absolute value.
//     C) Avoid cancellation errors when x is close to -1 and y is not large.
//     D) Avoid cancellation errors when x is close to -2 and y is not large.
//
//     Case A
//     ------
//
//     The real part of log1p reads:
//
//       0.5 * log((x + 1) ** 2 + y ** 2) = 0.5 * log1p(x + x + x * x + y * y)
//
//     When abs(y) < 1 and abs(x + 0.5 * y ** 2) is small, catastrophic
//     cancellation errors occur when evaluating `x + x + x * x + y * y`
//     using floating-point arithmetics. To avoid these errors, we'll use
//     Dekker's product for computing `x * x` and `y * y` which
//     effectively doubles the precision of the used floating-point
//     system. In addition, the terms are summed together using 2Sum
//     algorithm that minimizes cancellation errors. We'll have
//
//       xxh, xxl = square_dekker(x)
//       yyh, yyl = square_dekker(y)
//       x + x + x * x + y * y = sum_2sum([x + x, yyh, xxh, yyl, xxl])
//
//     which is accurate when the following inequalities hold:
//
//       abs(x) < sqrt(largest) * 0.1
//       abs(y) < sqrt(largest) * 0.99
//
//     [verified numerically for float32 and float64], except when x is
//     close to -1 (see Case C).
//
//     Case B
//     ------
//
//     If abs(x) or abs(y) is larger than sqrt(largest), squareing
//     these will overflow. To avoid such overflows, we'll apply
//     rescaling of log1p arguments.
//
//     First notice that if `abs(x) > sqrt(largest) > 4 / eps` holds then
//     `x + 1 ~= x`. Also, if `abs(x) < 4 / eps` then `(x + 1) ** 2 + y
//     ** 2 ~= y ** 2`. Proof:
//
//       (x + 1) ** 2 + y ** 2 ~= y ** 2    iff y ** 2 > 4 * (x + 1) ** 2 / eps
//
//       The lower limit to `y ** 2` is largest.  The upper limit to
//       `4 * (x + 1) ** 2 / eps` is `64 / eps ** 3` which is smaller than
//       largest. QED.
//
//     In conclusion, we can write
//
//       (x + 1) ** 2 + y ** 2 ~= x ** 2 + y ** 2
//
//     whenever abs(x) or abs(y) is greater than sqrt(largest).
//
//     Define
//
//       mx = max(abs(x), abs(y))
//       mn = min(abs(x), abs(y))
//
//     then under the given restrictions we'll have
//
//       real(log(x + I * y)) ~= 0.5 * log(x ** 2 + y ** 2)
//         = 0.5 * log(mx ** 2 * (1 + (mn / mx) ** 2))
//         = log(mx) + 0.5 * log1p((mn / mx) ** 2)
//
//     If mn == inf and mx == inf, we'll define `mn / mx == 1` for the
//     sake of reusing the above expression for complex infinities
//     (recall, `real(log(+-inf +-inf * I)) == inf`).
//
//     Case C
//     ------
//
//     If x is close to -1, then we'll use
//
//       real(log1p(x + I * y)) = 0.5 * log((1 + x) ** 2 + y ** 2)
//
//     which is accurate when the following inequalities hold:
//
//       -1.5 < x < -0.5  or  abs(x + 1) < 0.5
//       abs(y) < sqrt(largest)
//
//     [verified numerically for float32 and float64]. For simplicity,
//     we'll use the case C only when `abs(x) + abs(y) < 0.2`.
//
//     Case D
//     ------
//
//     If x is close to -2, the cancellation errors are avoided by using
//     the Case A method [verified numerically for float32 and float64].
//
//
def Log1pOp_ComplexElementType_ComplexMathExpander : Pat<(StableHLO_Log1pOp ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_MaxOp:$mx
         (StableHLO_AbsOp:$ax
           (StableHLO_RealOp:$x $z)),
         (StableHLO_AbsOp:$ay
           (StableHLO_ImagOp:$y $z))),
       (StableHLO_MulOp
         (StableHLO_SqrtOp
           (StableHLO_ConstantLikeMaxFiniteValue:$largest $x)),
         (StableHLO_ConstantLike<"0.01"> $x)),
        StableHLO_ComparisonDirectionValue<"GT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_AddOp
        (StableHLO_LogOp $mx),
        (StableHLO_MulOp
          (StableHLO_ConstantLike<"0.5">:$half $x),
          (StableHLO_Log1pOp
            (StableHLO_SelectOp
              (StableHLO_CompareOp
               (StableHLO_MinOp:$mn $ax, $ay),
               $mx,
                StableHLO_ComparisonDirectionValue<"EQ">,
                (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
              (StableHLO_ConstantLike<"1">:$one $x),
              (StableHLO_MulOp
                (StableHLO_DivOp:$r $mn, $mx),
                $r))))),
      (StableHLO_SelectOp
        (StableHLO_CompareOp
         (StableHLO_AddOp
           (StableHLO_AbsOp
             (StableHLO_AddOp:$xp1 $x, $one)),
           $ay),
         (StableHLO_ConstantLike<"0.2"> $x),
          StableHLO_ComparisonDirectionValue<"LT">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
        (StableHLO_MulOp
          $half,
          (StableHLO_LogOp
            (StableHLO_AddOp
              (StableHLO_MulOp $xp1, $xp1),
              (StableHLO_MulOp:$square_dekker_high $y, $y)))),
        (StableHLO_MulOp
          $half,
          (StableHLO_Log1pOp
            (StableHLO_AddOp:$sum_2sum_high
              (StableHLO_AddOp:$add_2sum_high
                (StableHLO_AddOp:$_add_2sum_high_0_
                  (StableHLO_AddOp:$_add_2sum_high_1_
                    (StableHLO_AddOp:$_add_2sum_high_2_
                      (StableHLO_AddOp:$x2h $x, $x),
                      $square_dekker_high),
                    (StableHLO_MulOp:$_square_dekker_high_0_ $x, $x)),
                  (StableHLO_AddOp:$square_dekker_low
                    (StableHLO_AddOp
                      (StableHLO_AddOp
                        (StableHLO_AddOp
                          (StableHLO_NegOp $square_dekker_high),
                          (StableHLO_MulOp
                            (StableHLO_AddOp:$yh
                              (StableHLO_MulOp:$multiply_veltkamp_splitter_constant_y
                                (StableHLO_SelectOp:$veltkamp_splitter_constant
                                  (StableHLO_CompareOp
                                   $largest,
                                   (StableHLO_ConstantLike<"1e+308"> $x),
                                    StableHLO_ComparisonDirectionValue<"GT">,
                                    (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                                  (StableHLO_ConstantLike<"134217729"> $x),
                                  (StableHLO_SelectOp
                                    (StableHLO_CompareOp
                                     $largest,
                                     (StableHLO_ConstantLike<"1e+38"> $x),
                                      StableHLO_ComparisonDirectionValue<"GT">,
                                      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                                    (StableHLO_ConstantLike<"4097"> $x),
                                    (StableHLO_ConstantLike<"65"> $x))),
                                $y),
                              (StableHLO_SubtractOp $y, $multiply_veltkamp_splitter_constant_y)),
                            $yh)),
                        (StableHLO_MulOp:$multiply_yh_yl
                          $yh,
                          (StableHLO_SubtractOp:$yl $y, $yh))),
                      $multiply_yh_yl),
                    (StableHLO_MulOp $yl, $yl))),
                (StableHLO_AddOp:$_square_dekker_low_0_
                  (StableHLO_AddOp
                    (StableHLO_AddOp
                      (StableHLO_AddOp
                        (StableHLO_NegOp $_square_dekker_high_0_),
                        (StableHLO_MulOp
                          (StableHLO_AddOp:$xh
                            (StableHLO_MulOp:$multiply_veltkamp_splitter_constant_x $veltkamp_splitter_constant, $x),
                            (StableHLO_SubtractOp $x, $multiply_veltkamp_splitter_constant_x)),
                          $xh)),
                      (StableHLO_MulOp:$multiply_xh_xl
                        $xh,
                        (StableHLO_SubtractOp:$xl $x, $xh))),
                    $multiply_xh_xl),
                  (StableHLO_MulOp $xl, $xl))),
              (StableHLO_AddOp
                (StableHLO_AddOp
                  (StableHLO_AddOp
                    (StableHLO_AddOp:$add_2sum_low
                      (StableHLO_SubtractOp
                        $x2h,
                        (StableHLO_SubtractOp
                          $_add_2sum_high_2_,
                          (StableHLO_SubtractOp:$subtract__add_2sum_high_2__x2h $_add_2sum_high_2_, $x2h))),
                      (StableHLO_SubtractOp $square_dekker_high, $subtract__add_2sum_high_2__x2h)),
                    (StableHLO_AddOp:$_add_2sum_low_0_
                      (StableHLO_SubtractOp
                        $_add_2sum_high_2_,
                        (StableHLO_SubtractOp
                          $_add_2sum_high_1_,
                          (StableHLO_SubtractOp:$subtract__add_2sum_high_1___add_2sum_high_2_ $_add_2sum_high_1_, $_add_2sum_high_2_))),
                      (StableHLO_SubtractOp $_square_dekker_high_0_, $subtract__add_2sum_high_1___add_2sum_high_2_))),
                  (StableHLO_AddOp:$_add_2sum_low_1_
                    (StableHLO_SubtractOp
                      $_add_2sum_high_1_,
                      (StableHLO_SubtractOp
                        $_add_2sum_high_0_,
                        (StableHLO_SubtractOp:$subtract__add_2sum_high_0___add_2sum_high_1_ $_add_2sum_high_0_, $_add_2sum_high_1_))),
                    (StableHLO_SubtractOp $square_dekker_low, $subtract__add_2sum_high_0___add_2sum_high_1_))),
                (StableHLO_AddOp:$_add_2sum_low_2_
                  (StableHLO_SubtractOp
                    $_add_2sum_high_0_,
                    (StableHLO_SubtractOp
                      $add_2sum_high,
                      (StableHLO_SubtractOp:$subtract_add_2sum_high__add_2sum_high_0_ $add_2sum_high, $_add_2sum_high_0_))),
                  (StableHLO_SubtractOp $_square_dekker_low_0_, $subtract_add_2sum_high__add_2sum_high_0_)))))))),
    (StableHLO_Atan2Op $y, $xp1))>;
