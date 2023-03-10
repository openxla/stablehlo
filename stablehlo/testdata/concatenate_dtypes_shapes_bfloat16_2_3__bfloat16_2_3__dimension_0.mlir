// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xbf16>, tensor<2x3xbf16>)
    %1 = call @expected() : () -> tensor<4x3xbf16>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 0 : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<4x3xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x3xbf16>, tensor<4x3xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xbf16>, tensor<2x3xbf16>) {
    %0 = stablehlo.constant dense<[[3.390630e+00, -1.882810e+00, -1.953130e-01], [3.781250e+00, -3.531250e+00, -4.812500e+00]]> : tensor<2x3xbf16>
    %1 = stablehlo.constant dense<[[-1.585940e+00, -5.375000e+00, -2.359380e+00], [2.296880e+00, -1.992190e+00, -4.250000e+00]]> : tensor<2x3xbf16>
    return %0, %1 : tensor<2x3xbf16>, tensor<2x3xbf16>
  }
  func.func private @expected() -> tensor<4x3xbf16> {
    %0 = stablehlo.constant dense<[[3.390630e+00, -1.882810e+00, -1.953130e-01], [3.781250e+00, -3.531250e+00, -4.812500e+00], [-1.585940e+00, -5.375000e+00, -2.359380e+00], [2.296880e+00, -1.992190e+00, -4.250000e+00]]> : tensor<4x3xbf16>
    return %0 : tensor<4x3xbf16>
  }
}
