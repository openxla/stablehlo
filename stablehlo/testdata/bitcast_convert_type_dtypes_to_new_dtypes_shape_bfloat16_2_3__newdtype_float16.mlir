// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xbf16>
    %1 = call @expected() : () -> tensor<2x3xf16>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xbf16>) -> tensor<2x3xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[3.953130e+00, -5.976560e-01, -3.300780e-01], [-2.765630e+00, -7.625000e+00, 9.531250e-01]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
  func.func private @expected() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<[[2.244140e+00, -1.774410e+00, -1.665040e+00], [-2.095700e+00, -2.476560e+00, 1.863280e+00]]> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
}
