// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x5xbf16>
    %1 = call @expected() : () -> tensor<4x5xbf16>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x5xbf16>, tensor<4x5xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x5xbf16> {
    %0 = stablehlo.constant dense<[[-2.843750e+00, -4.281250e+00, 2.937500e+00, -5.000000e+00, -1.101560e+00], [1.765630e+00, 7.812500e-01, -2.984380e+00, 2.312500e+00, -4.453130e-01], [-6.484380e-01, 1.929690e+00, 3.964840e-01, 1.320310e+00, -2.343750e+00], [-1.656250e+00, 5.742190e-01, -1.179690e+00, -4.750000e+00, 7.617180e-02]]> : tensor<4x5xbf16>
    return %0 : tensor<4x5xbf16>
  }
  func.func private @expected() -> tensor<4x5xbf16> {
    %0 = stablehlo.constant dense<[[-1.656250e+00, 5.742190e-01, -1.179690e+00, -4.750000e+00, 7.617180e-02], [-6.484380e-01, 1.929690e+00, 3.964840e-01, 1.320310e+00, -2.343750e+00], [1.765630e+00, 7.812500e-01, -2.984380e+00, 2.312500e+00, -4.453130e-01], [-2.843750e+00, -4.281250e+00, 2.937500e+00, -5.000000e+00, -1.101560e+00]]> : tensor<4x5xbf16>
    return %0 : tensor<4x5xbf16>
  }
}
