// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4xf16>
    %1 = call @expected() : () -> tensor<3x4xf16>
    %2 = stablehlo.custom_call @check.eq(%0, %1) : (tensor<3x4xf16>, tensor<3x4xf16>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4xf16> {
    %0 = stablehlo.constant dense<[[-1.666020e+00, 3.050780e+00, 2.912110e+00, -4.992190e+00], [-3.333980e+00, -1.374020e+00, -1.397460e+00, 2.365230e+00], [3.532710e-01, -1.795900e+00, -1.781250e+00, 1.457030e+00]]> : tensor<3x4xf16>
    return %0 : tensor<3x4xf16>
  }
  func.func private @expected() -> tensor<3x4xf16> {
    %0 = stablehlo.constant dense<[[-1.666020e+00, 3.050780e+00, 2.912110e+00, -4.992190e+00], [-3.333980e+00, -1.374020e+00, -1.397460e+00, 2.365230e+00], [3.532710e-01, -1.795900e+00, -1.781250e+00, 1.457030e+00]]> : tensor<3x4xf16>
    return %0 : tensor<3x4xf16>
  }
}
