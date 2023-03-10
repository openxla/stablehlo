// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xbf16>
    %1 = call @expected() : () -> tensor<2x3xi16>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xbf16>) -> tensor<2x3xi16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[-1.570310e+00, -1.718750e+00, -1.347660e-01], [-1.210940e+00, -3.484380e+00, 2.406250e+00]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
  func.func private @expected() -> tensor<2x3xi16> {
    %0 = stablehlo.constant dense<[[-16439, -16420, -16886], [-16485, -16289, 16410]]> : tensor<2x3xi16>
    return %0 : tensor<2x3xi16>
  }
}
