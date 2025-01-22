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
// This file is generated using functional_algorithms tool (0.15.0).
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

def StableHLO_ConstantLikePosInfValue : NativeCodeCall<
    "::mlir::stablehlo::getConstantLikeInfValue($_builder, $_loc, $0, /*negative=*/false)">;
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
def Log1pOp_ComplexElementType_ComplexMathExpander: Pat<(StableHLO_Log1pOp ComplexElementType:$z),
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

// Square root on complex inputs:
//
//       sqrt(z) = sqrt((hypot(x, y) + x)/2) + I * sgn(y) * sqrt((hypot(x, y) - x) / 2)
//
//     where z = x + I * y, sgn(y) = 1 if y >= 0, and sgn(y) = -1 otherwise.
//
//     Algorithm
//     ---------
//
//     In the above formula, catastrophic cancellation errors occur in
//     the imaginary part when x is positive, and in the real part when x
//     is negative. To avoid these, let us define
//
//       u = sqrt((hypot(x, y) + abs(x))/2)
//       v = sgn(y) * sqrt((hypot(x, y) - abs(x))/2)
//
//     and find
//
//       u * v = sgn(y) * sqrt(hypot(x, y) ** 2 - x ** 2) / 2 = y / 2
//
//     That is, if x > 0, then we have
//
//       sqrt(z) = u + I * y / u / 2
//
//     and if x < 0,
//
//       sqrt(z) = abs(y) / u / 2 + I * sgn(y) * u
//
//     If abs(x) and abs(y) are smaller that smallest normal, then as a
//     result of underflow, u will be zero and v will be undefined. On
//     the other hand, if abs(x) and abs(y) are close to largest floating
//     point number, then `hypot(x, y) + abs(x)` will overflow, and u
//     will be `inf`. To address the issues from underflow and overflow,
//     we'll use the following formula:
//
//     1. abs(x) == abs(y), or abs(x) == inf and abs(y) == inf, then
//
//       u_eq = sqrt(abs(x)) * sqrt((1 + sqrt(2))/2)
//       abs(y) / u = sqrt(abs(x)) / sqrt((1 + sqrt(2))/2)
//
//     2. If abs(x) > abs(y) and u == 0 (the underflow case) or u == inf
//       (the overflow case), denote r = abs(y) / abs(x), then
//
//       u_gt = sqrt(abs(x)) * sqrt((1 + hypot(1, r)) / 2)
//       abs(y) / u = sqrt(abs(y)) * sqrt(r) / sqrt((1 + hypot(1, r)) / 2)
//
//     3. If abs(x) < abs(y) and u == 0 (the underflow case) or u == inf
//       (the overflow case), denote r = abs(x) / abs(y), then
//
//       u_lt = sqrt(abs(y)) * sqrt((r + sqrt(1, r)) / 2)
//       abs(y) / u = sqrt(abs(y)) / sqrt((r + sqrt(1, r)) / 2)
//
def SqrtOp_ComplexElementType_ComplexMathExpander: Pat<(StableHLO_SqrtOp ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_RealOp:$x $z),
       (StableHLO_ConstantLike<"0">:$constant_0 $x),
        StableHLO_ComparisonDirectionValue<"GE">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_SelectOp:$u
        (StableHLO_CompareOp:$eq_ax_ay
         (StableHLO_AbsOp:$ax $x),
         (StableHLO_AbsOp:$ay
           (StableHLO_ImagOp:$y $z)),
          StableHLO_ComparisonDirectionValue<"EQ">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
        (StableHLO_DivOp
          (StableHLO_MulOp
            (StableHLO_SqrtOp:$sq_ax $ax),
            (StableHLO_ConstantLike<"1.5537739740300374"> $x)),
          (StableHLO_ConstantLike<"1.4142135623730951">:$sq_2 $x)),
        (StableHLO_SelectOp
          (StableHLO_OrOp:$logical_or_eq_u_general_constant_0_eq_u_general_constant_posinf
            (StableHLO_CompareOp
             (StableHLO_SqrtOp:$u_general
               (StableHLO_AddOp
                 (StableHLO_DivOp
                   (StableHLO_SelectOp
                     (StableHLO_CompareOp
                      (StableHLO_MaxOp:$mx $ax, $ay),
                      (StableHLO_MinOp:$mn $ax, $ay),
                       StableHLO_ComparisonDirectionValue<"EQ">,
                       (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                     (StableHLO_MulOp $sq_2, $mx),
                     (StableHLO_SelectOp
                       (StableHLO_AndOp
                         (StableHLO_CompareOp
                          (StableHLO_SqrtOp:$sqa
                            (StableHLO_AddOp
                              (StableHLO_ConstantLike<"1">:$one $x),
                              (StableHLO_MulOp:$r
                                (StableHLO_DivOp:$mn_over_mx $mn, $mx),
                                $mn_over_mx))),
                          $one,
                           StableHLO_ComparisonDirectionValue<"EQ">,
                           (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                         (StableHLO_CompareOp
                          $r,
                          $constant_0,
                           StableHLO_ComparisonDirectionValue<"GT">,
                           (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
                       (StableHLO_AddOp
                         $mx,
                         (StableHLO_DivOp
                           (StableHLO_MulOp $mx, $r),
                           (StableHLO_ConstantLike<"2">:$two $x))),
                       (StableHLO_MulOp $mx, $sqa))),
                   $two),
                 (StableHLO_DivOp $ax, $two))),
             $constant_0,
              StableHLO_ComparisonDirectionValue<"EQ">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            (StableHLO_CompareOp
             $u_general,
             (StableHLO_ConstantLikePosInfValue $x),
              StableHLO_ComparisonDirectionValue<"EQ">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
          (StableHLO_SelectOp
            (StableHLO_CompareOp:$gt_ax_ay
             $ax,
             $ay,
              StableHLO_ComparisonDirectionValue<"GT">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            (StableHLO_MulOp
              $sq_ax,
              (StableHLO_DivOp
                (StableHLO_SqrtOp:$sq_1h
                  (StableHLO_AddOp
                    $one,
                    (StableHLO_SelectOp:$h
                      (StableHLO_CompareOp
                       (StableHLO_MaxOp:$_mx_0_
                         $one,
                         (StableHLO_AbsOp:$abs__r_0_
                           (StableHLO_SelectOp:$_r_0_
                             $eq_ax_ay,
                             $one,
                             (StableHLO_SelectOp
                               (StableHLO_CompareOp:$lt_ax_ay
                                $ax,
                                $ay,
                                 StableHLO_ComparisonDirectionValue<"LT">,
                                 (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                               (StableHLO_DivOp $ax, $ay),
                               (StableHLO_DivOp $ay, $ax))))),
                       (StableHLO_MinOp:$_mn_0_ $one, $abs__r_0_),
                        StableHLO_ComparisonDirectionValue<"EQ">,
                        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                      (StableHLO_MulOp $sq_2, $_mx_0_),
                      (StableHLO_SelectOp
                        (StableHLO_AndOp
                          (StableHLO_CompareOp
                           (StableHLO_SqrtOp:$_sqa_0_
                             (StableHLO_AddOp
                               $one,
                               (StableHLO_MulOp:$_r_1_
                                 (StableHLO_DivOp:$_mn_over_mx_0_ $_mn_0_, $_mx_0_),
                                 $_mn_over_mx_0_))),
                           $one,
                            StableHLO_ComparisonDirectionValue<"EQ">,
                            (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                          (StableHLO_CompareOp
                           $_r_1_,
                           $constant_0,
                            StableHLO_ComparisonDirectionValue<"GT">,
                            (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
                        (StableHLO_AddOp
                          $_mx_0_,
                          (StableHLO_DivOp
                            (StableHLO_MulOp $_mx_0_, $_r_1_),
                            $two)),
                        (StableHLO_MulOp $_mx_0_, $_sqa_0_))))),
                $sq_2)),
            (StableHLO_MulOp
              (StableHLO_SqrtOp:$sq_ay $ay),
              (StableHLO_DivOp
                (StableHLO_SqrtOp:$sq_rh
                  (StableHLO_AddOp $_r_0_, $h)),
                $sq_2))),
          $u_general)),
      (StableHLO_SelectOp:$ay_div_u
        $eq_ax_ay,
        (StableHLO_DivOp
          $sq_ay,
          (StableHLO_ConstantLike<"2.19736822693562"> $x)),
        (StableHLO_SelectOp
          $logical_or_eq_u_general_constant_0_eq_u_general_constant_posinf,
          (StableHLO_SelectOp
            $gt_ax_ay,
            (StableHLO_DivOp
              (StableHLO_MulOp
                $sq_ay,
                (StableHLO_SelectOp
                  $eq_ax_ay,
                  $one,
                  (StableHLO_SelectOp
                    $lt_ax_ay,
                    (StableHLO_DivOp $sq_ax, $sq_ay),
                    (StableHLO_DivOp $sq_ay, $sq_ax)))),
              (StableHLO_MulOp $sq_1h, $sq_2)),
            (StableHLO_DivOp
              $sq_ay,
              (StableHLO_MulOp $sq_rh, $sq_2))),
          (StableHLO_DivOp
            $ay,
            (StableHLO_MulOp $u_general, $two))))),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       $x,
       $constant_0,
        StableHLO_ComparisonDirectionValue<"LT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_SelectOp
        (StableHLO_CompareOp:$lt_y_constant_0
         $y,
         $constant_0,
          StableHLO_ComparisonDirectionValue<"LT">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
        (StableHLO_NegOp $u),
        $u),
      (StableHLO_SelectOp
        $lt_y_constant_0,
        (StableHLO_NegOp $ay_div_u),
        $ay_div_u)))>;

// Logarithm of z on complex input:
//
//       log(x + I * y) = log(hypot(x, y)) + I * arctan2(y, x)
//
//     where
//
//       x and y are real and imaginary parts of the input to log1p, and
//       I is imaginary unit.
//
//     For evaluating the real part of log accurately on the whole
//     complex plane, we'll use the following formula:
//
//     Case A
//     ------
//
//     If `hypot(x, y)` is close to one, we'll use Dekker product and
//     2Sum algorithm to double the precision of computations to minimize
//     cancellation errors:
//
//       xxh, xxl = square_dekker(x)
//       yyh, yyl = square_dekker(y)
//       s = -1 + x * x + y * y = sum_2sum([-1, yyh, xxh, yyl, xxl])
//
//     so that
//
//       real(log(x + I * y)) = 1/2 * log1p(s)
//
//     when abs(s) < 0.5.
//
//     Case B
//     ------
//
//     Otherwise, we'll use
//
//       real(log(x + I * y)) = log(mx) + 1/2 * log1p((mn / mx) ** 2)
//
//     where
//
//       mx = max(abs(x), abs(y))
//       mn = min(abs(x), abs(y))
//
//     For `mx == mn == inf` or `mx == mn == 0` case, we'll define `mn / mx == 1`.
//
//
def LogOp_ComplexElementType_ComplexMathExpander: Pat<(StableHLO_LogOp ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_AbsOp
         (StableHLO_AddOp:$sum_fast2sum_high
           (StableHLO_AddOp:$add_fast2sum_high
             (StableHLO_AddOp:$_add_fast2sum_high_0_
               (StableHLO_AddOp:$_add_fast2sum_high_1_
                 (StableHLO_AddOp:$_add_fast2sum_high_2_
                   (StableHLO_ConstantLike<"-1.0">:$constant_fneg1 (StableHLO_RealOp:$x $z)),
                   (StableHLO_SelectOp:$mxh
                     (StableHLO_CompareOp:$gt_square_dekker_high__square_dekker_high_0_
                      (StableHLO_MulOp:$square_dekker_high
                        (StableHLO_ImagOp:$y $z),
                        $y),
                      (StableHLO_MulOp:$_square_dekker_high_0_ $x, $x),
                       StableHLO_ComparisonDirectionValue<"GT">,
                       (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                     $square_dekker_high,
                     $_square_dekker_high_0_)),
                 (StableHLO_SelectOp:$mnh $gt_square_dekker_high__square_dekker_high_0_, $_square_dekker_high_0_, $square_dekker_high)),
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
                                (StableHLO_ConstantLikeMaxFiniteValue:$largest $x),
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
                 (StableHLO_SubtractOp:$add_fast2sum_low
                   $mxh,
                   (StableHLO_SubtractOp $_add_fast2sum_high_2_, $constant_fneg1)),
                 (StableHLO_SubtractOp:$_add_fast2sum_low_0_
                   $mnh,
                   (StableHLO_SubtractOp $_add_fast2sum_high_1_, $_add_fast2sum_high_2_))),
               (StableHLO_SubtractOp:$_add_fast2sum_low_1_
                 $square_dekker_low,
                 (StableHLO_SubtractOp $_add_fast2sum_high_0_, $_add_fast2sum_high_1_))),
             (StableHLO_SubtractOp:$_add_fast2sum_low_2_
               $_square_dekker_low_0_,
               (StableHLO_SubtractOp $add_fast2sum_high, $_add_fast2sum_high_0_))))),
       (StableHLO_ConstantLike<"0.5">:$half $x),
        StableHLO_ComparisonDirectionValue<"LT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_MulOp
        $half,
        (StableHLO_Log1pOp $sum_fast2sum_high)),
      (StableHLO_AddOp
        (StableHLO_LogOp
          (StableHLO_MaxOp:$mx
            (StableHLO_AbsOp:$abs_x $x),
            (StableHLO_AbsOp:$abs_y $y))),
        (StableHLO_MulOp
          $half,
          (StableHLO_Log1pOp
            (StableHLO_MulOp
              (StableHLO_SelectOp:$mn_over_mx
                (StableHLO_CompareOp
                 (StableHLO_MinOp:$mn $abs_x, $abs_y),
                 $mx,
                  StableHLO_ComparisonDirectionValue<"EQ">,
                  (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                (StableHLO_ConstantLike<"1.0"> $x),
                (StableHLO_DivOp $mn, $mx)),
              $mn_over_mx))))),
    (StableHLO_Atan2Op $y, $x))>;

// Exponential on complex inputs:
//
//       exp(z) = exp(x) * (cos(y) + I * sin(y))
//
//     where z = x + I * y.
//
//     Algorithm
//     ---------
//
//     While the above expression is accurate for a large part of the
//     complex plane, there is two cases that require special attention.
//
//     First, when `y == 0`, we'll define
//
//       imag(exp(z)) = 0
//
//     that otherwise for overflowing `exp(x)` would evaluate to nan.
//
//     Second, the overflow case `exp(x) -> inf` is compensated when `y`
//     is close to the zeros of `cos(y)` or `sin(y)` and the real or
//     imaginary parts of `exp(z)` ought to be finite. Therefore, for the
//     `exp(x) -> inf` case, we'll use
//
//       exp(z) = exp(x / 2) * (cos(y) + I * sin(y)) * exp(x / 2)
//
//     Notice that for `y != 0`, neither `cos(y)` nor `sin(y)` is never
//     zero on the set of floating point numbers.
//
def ConstDefaultResultAccuracyAttr :
  ConstantAttr<StableHLO_ResultAccuracyAttr, "::mlir::stablehlo::ResultAccuracyMode::DEFAULT">;

def ExpOp_ComplexElementType_ComplexMathExpander: Pat<(StableHLO_ExpOp ComplexElementType:$z, ConstDefaultResultAccuracyAttr),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp:$eq_e_constant_posinf
       (StableHLO_ExpOp:$e
         (StableHLO_RealOp:$x $z), ConstDefaultResultAccuracyAttr),
       (StableHLO_ConstantLikePosInfValue $x),
        StableHLO_ComparisonDirectionValue<"EQ">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_MulOp
        (StableHLO_MulOp
          (StableHLO_ExpOp:$e2
            (StableHLO_MulOp
              $x,
              (StableHLO_ConstantLike<"0.5"> $x)), ConstDefaultResultAccuracyAttr),
          (StableHLO_CosineOp:$cs
            (StableHLO_ImagOp:$y $z))),
        $e2),
      (StableHLO_MulOp $e, $cs)),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       $y,
       (StableHLO_ConstantLike<"0">:$zero $x),
        StableHLO_ComparisonDirectionValue<"EQ">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      $zero,
      (StableHLO_SelectOp
        $eq_e_constant_posinf,
        (StableHLO_MulOp
          (StableHLO_MulOp
            $e2,
            (StableHLO_SineOp:$sn $y)),
          $e2),
        (StableHLO_MulOp $e, $sn))))>;
