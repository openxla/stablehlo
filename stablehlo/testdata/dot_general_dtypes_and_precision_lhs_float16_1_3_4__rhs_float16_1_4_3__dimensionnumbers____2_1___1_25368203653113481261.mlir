// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xf16>, tensor<1x4x3xf16>)
    %1 = call @expected() : () -> tensor<1xf16>
    %2 = stablehlo.convert %0#0 : (tensor<1x3x4xf16>) -> tensor<1x3x4xf32>
    %3 = stablehlo.convert %0#1 : (tensor<1x4x3xf16>) -> tensor<1x4x3xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<1x3x4xf32>, tensor<1x4x3xf32>) -> tensor<1xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<1xf16>, tensor<1xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xf16>, tensor<1x4x3xf16>) {
    %0 = stablehlo.constant dense<[[[-1.402340e+00, 5.394530e+00, 2.120970e-03, 2.400390e+00], [-3.042970e+00, 1.202150e+00, -3.568360e+00, -3.050780e+00], [-4.812620e-02, -4.140630e+00, 4.054690e+00, 2.460940e+00]]]> : tensor<1x3x4xf16>
    %1 = stablehlo.constant dense<[[[5.445310e+00, -3.890630e+00, 1.907230e+00], [-1.224610e+00, 6.804680e+00, 8.696280e-01], [-2.195310e+00, -3.376950e+00, -8.461910e-01], [-4.714840e+00, 1.102540e+00, 5.871090e+00]]]> : tensor<1x4x3xf16>
    return %0, %1 : tensor<1x3x4xf16>, tensor<1x4x3xf16>
  }
  func.func private @expected() -> tensor<1xf16> {
    %0 = stablehlo.constant dense<1.046880e+01> : tensor<1xf16>
    return %0 : tensor<1xf16>
  }
}
