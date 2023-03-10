// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4xbf16>
    %1 = call @expected() : () -> tensor<3x4xbf16>
    %2 = stablehlo.custom_call @check.eq(%0, %1) : (tensor<3x4xbf16>, tensor<3x4xbf16>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4xbf16> {
    %0 = stablehlo.constant dense<[[9.765620e-01, 2.734380e+00, 4.468750e+00, -2.490230e-01], [-2.558590e-01, 3.140630e+00, -2.859380e+00, -1.992190e+00], [1.046880e+00, 1.375000e+00, 3.656250e+00, -1.336670e-02]]> : tensor<3x4xbf16>
    return %0 : tensor<3x4xbf16>
  }
  func.func private @expected() -> tensor<3x4xbf16> {
    %0 = stablehlo.constant dense<[[9.765620e-01, 2.734380e+00, 4.468750e+00, -2.490230e-01], [-2.558590e-01, 3.140630e+00, -2.859380e+00, -1.992190e+00], [1.046880e+00, 1.375000e+00, 3.656250e+00, -1.336670e-02]]> : tensor<3x4xbf16>
    return %0 : tensor<3x4xbf16>
  }
}
