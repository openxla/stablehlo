// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xbf16>, tensor<1x4x3xbf16>)
    %1 = call @expected() : () -> tensor<1xbf16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<1x3x4xbf16>, tensor<1x4x3xbf16>) -> tensor<1xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xbf16>, tensor<1x4x3xbf16>) {
    %0 = stablehlo.constant dense<[[[-1.703130e+00, -5.375000e+00, -3.203130e+00, -1.867190e+00], [5.312500e+00, -5.703130e-01, -2.890630e-01, 4.589840e-01], [-2.281250e+00, 1.898440e+00, -6.367190e-01, 9.500000e+00]]]> : tensor<1x3x4xbf16>
    %1 = stablehlo.constant dense<[[[-4.906250e+00, -6.156250e+00, 7.929680e-01], [3.328130e+00, 1.101560e+00, 6.562500e-01], [-1.609380e+00, -2.281250e+00, -9.804680e-01], [7.851560e-01, -1.132810e-01, 5.843750e+00]]]> : tensor<1x4x3xbf16>
    return %0, %1 : tensor<1x3x4xbf16>, tensor<1x4x3xbf16>
  }
  func.func private @expected() -> tensor<1xbf16> {
    %0 = stablehlo.constant dense<1.700000e+01> : tensor<1xbf16>
    return %0 : tensor<1xbf16>
  }
}
