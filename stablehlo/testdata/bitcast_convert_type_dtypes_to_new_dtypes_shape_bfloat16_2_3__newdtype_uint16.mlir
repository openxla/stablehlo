// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xbf16>
    %1 = call @expected() : () -> tensor<2x3xui16>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xbf16>) -> tensor<2x3xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xui16>, tensor<2x3xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[2.718750e+00, -4.093750e+00, -6.484380e-01], [-1.203130e+00, -1.923830e-01, 3.066410e-01]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
  func.func private @expected() -> tensor<2x3xui16> {
    %0 = stablehlo.constant dense<[[16430, 49283, 48934], [49050, 48709, 16029]]> : tensor<2x3xui16>
    return %0 : tensor<2x3xui16>
  }
}
