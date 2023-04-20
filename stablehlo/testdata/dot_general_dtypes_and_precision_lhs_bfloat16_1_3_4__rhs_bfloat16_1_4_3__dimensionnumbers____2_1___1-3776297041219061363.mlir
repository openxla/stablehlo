// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xbf16>, tensor<1x4x3xbf16>)
    %1 = call @expected() : () -> tensor<1xbf16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<1x3x4xbf16>, tensor<1x4x3xbf16>) -> tensor<1xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xbf16>, tensor<1x4x3xbf16>) {
    %0 = stablehlo.constant dense<[[[-3.968750e+00, 1.140630e+00, 7.812500e-01, -1.437500e+00], [-3.250000e+00, -4.648440e-01, 4.281250e+00, -6.562500e+00], [-3.343750e+00, 2.753910e-01, -5.812500e+00, 3.125000e+00]]]> : tensor<1x3x4xbf16>
    %1 = stablehlo.constant dense<[[[-2.562500e+00, -3.328130e+00, -2.828130e+00], [-2.093750e+00, -2.500000e-01, -3.281250e+00], [-3.062500e+00, 8.515620e-01, -1.140630e+00], [-4.437500e+00, 3.250000e+00, -3.921880e+00]]]> : tensor<1x4x3xbf16>
    return %0, %1 : tensor<1x3x4xbf16>, tensor<1x4x3xbf16>
  }
  func.func private @expected() -> tensor<1xbf16> {
    %0 = stablehlo.constant dense<7.937500e+00> : tensor<1xbf16>
    return %0 : tensor<1xbf16>
  }
}
