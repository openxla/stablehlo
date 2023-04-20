// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xbf16>
    %1 = call @expected() : () -> tensor<2x3xbf16>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[8.515620e-01, 9.570310e-01, 6.125000e+00], [2.671880e+00, -6.484380e-01, -3.031250e+00]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
  func.func private @expected() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[8.515620e-01, 9.570310e-01, 6.125000e+00], [2.671880e+00, -6.484380e-01, -3.031250e+00]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
}
