// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xf16>
    %1 = call @expected() : () -> tensor<2x3xui16>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xf16>) -> tensor<2x3xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xui16>, tensor<2x3xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<[[4.386720e+00, -9.877920e-01, 1.872070e+00], [4.785160e-01, 3.804690e+00, -2.134770e+00]]> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
  func.func private @expected() -> tensor<2x3xui16> {
    %0 = stablehlo.constant dense<[[17507, 48103, 16253], [14248, 17308, 49221]]> : tensor<2x3xui16>
    return %0 : tensor<2x3xui16>
  }
}
