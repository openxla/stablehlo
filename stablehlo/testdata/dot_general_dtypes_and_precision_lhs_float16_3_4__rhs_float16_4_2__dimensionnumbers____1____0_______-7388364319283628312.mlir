// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xf16>, tensor<4x2xf16>)
    %1 = call @expected() : () -> tensor<3x2xf16>
    %2 = stablehlo.convert %0#0 : (tensor<3x4xf16>) -> tensor<3x4xf32>
    %3 = stablehlo.convert %0#1 : (tensor<4x2xf16>) -> tensor<4x2xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x2xf16>, tensor<3x2xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xf16>, tensor<4x2xf16>) {
    %0 = stablehlo.constant dense<[[1.265630e+00, -1.868160e+00, 1.634770e+00, -7.075190e-01], [-2.451170e+00, -7.929680e-01, -7.080080e-01, -2.072270e+00], [5.102540e-01, -3.515630e+00, -1.195310e+00, -3.314450e+00]]> : tensor<3x4xf16>
    %1 = stablehlo.constant dense<[[-1.906250e+00, -6.186520e-01], [3.503910e+00, 6.589840e+00], [-6.136710e+00, 5.083010e-01], [-4.113280e+00, 2.857420e+00]]> : tensor<4x2xf16>
    return %0, %1 : tensor<3x4xf16>, tensor<4x2xf16>
  }
  func.func private @expected() -> tensor<3x2xf16> {
    %0 = stablehlo.constant dense<[[-1.607810e+01, -1.428130e+01], [1.476560e+01, -9.992180e+00], [7.675780e+00, -3.356250e+01]]> : tensor<3x2xf16>
    return %0 : tensor<3x2xf16>
  }
}
