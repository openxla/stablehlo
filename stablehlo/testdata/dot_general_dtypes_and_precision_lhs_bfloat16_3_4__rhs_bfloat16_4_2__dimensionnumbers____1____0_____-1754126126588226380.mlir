// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xbf16>, tensor<4x2xbf16>)
    %1 = call @expected() : () -> tensor<3x2xbf16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<3x4xbf16>, tensor<4x2xbf16>) -> tensor<3x2xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xbf16>, tensor<3x2xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xbf16>, tensor<4x2xbf16>) {
    %0 = stablehlo.constant dense<[[9.101560e-01, -3.125000e-01, -6.000000e+00, 2.421880e+00], [2.625000e+00, 3.968750e+00, -4.187500e+00, 2.390630e+00], [4.375000e+00, 4.093750e+00, 9.023430e-01, 8.398430e-02]]> : tensor<3x4xbf16>
    %1 = stablehlo.constant dense<[[-1.687500e+00, 4.625000e+00], [3.546880e+00, -1.826170e-01], [-3.312500e+00, -1.421880e+00], [-4.031250e+00, 2.988280e-01]]> : tensor<4x2xbf16>
    return %0, %1 : tensor<3x4xbf16>, tensor<4x2xbf16>
  }
  func.func private @expected() -> tensor<3x2xbf16> {
    %0 = stablehlo.constant dense<[[7.468750e+00, 1.350000e+01], [1.387500e+01, 1.812500e+01], [3.812500e+00, 1.825000e+01]]> : tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}
