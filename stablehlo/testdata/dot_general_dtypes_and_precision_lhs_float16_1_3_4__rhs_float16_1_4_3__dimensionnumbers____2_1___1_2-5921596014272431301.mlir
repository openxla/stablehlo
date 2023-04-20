// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xf16>, tensor<1x4x3xf16>)
    %1 = call @expected() : () -> tensor<1xf16>
    %2 = stablehlo.convert %0#0 : (tensor<1x3x4xf16>) -> tensor<1x3x4xf32>
    %3 = stablehlo.convert %0#1 : (tensor<1x4x3xf16>) -> tensor<1x4x3xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<1x3x4xf32>, tensor<1x4x3xf32>) -> tensor<1xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<1xf16>, tensor<1xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xf16>, tensor<1x4x3xf16>) {
    %0 = stablehlo.constant dense<[[[-2.099610e+00, 3.333980e+00, -1.626950e+00, 2.269530e+00], [1.793950e+00, -4.488280e+00, -2.775390e+00, 3.843750e+00], [1.340820e+00, 5.433590e+00, 2.652340e+00, -2.703130e+00]]]> : tensor<1x3x4xf16>
    %1 = stablehlo.constant dense<[[[6.542960e-01, 1.103520e+00, 2.219240e-01], [2.421880e+00, -6.234380e+00, -4.015630e+00], [-1.750980e+00, 6.835930e-01, -1.810550e+00], [-3.034970e-02, 3.303220e-01, -9.907220e-01]]]> : tensor<1x4x3xf16>
    return %0, %1 : tensor<1x3x4xf16>, tensor<1x4x3xf16>
  }
  func.func private @expected() -> tensor<1xf16> {
    %0 = stablehlo.constant dense<1.517190e+01> : tensor<1xf16>
    return %0 : tensor<1xf16>
  }
}
