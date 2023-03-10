// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xbf16>, tensor<4x2xbf16>)
    %1 = call @expected() : () -> tensor<3x2xbf16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<3x4xbf16>, tensor<4x2xbf16>) -> tensor<3x2xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xbf16>, tensor<3x2xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xbf16>, tensor<4x2xbf16>) {
    %0 = stablehlo.constant dense<[[1.296880e+00, -2.343750e+00, -4.031250e+00, 5.625000e-01], [1.687500e+00, -2.390630e+00, -5.812500e+00, -2.656250e+00], [-4.500000e+00, -1.892090e-02, -2.453130e+00, 1.406250e+00]]> : tensor<3x4xbf16>
    %1 = stablehlo.constant dense<[[-6.640630e-01, 8.937500e+00], [3.906250e-01, -3.613280e-01], [3.375000e+00, 1.945310e+00], [1.117190e+00, -2.640630e+00]]> : tensor<4x2xbf16>
    return %0, %1 : tensor<3x4xbf16>, tensor<4x2xbf16>
  }
  func.func private @expected() -> tensor<3x2xbf16> {
    %0 = stablehlo.constant dense<[[-1.475000e+01, 3.109380e+00], [-2.462500e+01, 1.162500e+01], [-3.734380e+00, -4.875000e+01]]> : tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}
