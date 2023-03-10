// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf16>
    %1 = call @expected() : () -> tensor<3x5xf16>
    %2 = stablehlo.constant dense<0x7C00> : tensor<f16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f16>) -> tensor<f16>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %6 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %6 : tensor<f16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xf16>, tensor<3x5xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf16> {
    %0 = stablehlo.constant dense<[[-5.363280e+00, 1.365230e+00, 1.684570e+00, -2.314450e+00, 6.393430e-03, -2.237550e-01], [3.558590e+00, 1.348630e+00, -2.451170e+00, -2.970700e+00, -2.535160e+00, 7.402340e+00], [-1.527100e-01, 1.347660e+00, -3.291020e+00, -3.820310e+00, 5.693360e-01, -1.842770e+00], [2.046880e+00, 8.798820e-01, -1.811520e+00, -2.761720e+00, -3.400390e+00, 3.554690e-01]]> : tensor<4x6xf16>
    return %0 : tensor<4x6xf16>
  }
  func.func private @expected() -> tensor<3x5xf16> {
    %0 = stablehlo.constant dense<[[-5.363280e+00, -2.451170e+00, -2.970700e+00, -2.970700e+00, -2.535160e+00], [-1.527100e-01, -3.291020e+00, -3.820310e+00, -3.820310e+00, -2.535160e+00], [-1.527100e-01, -3.291020e+00, -3.820310e+00, -3.820310e+00, -3.400390e+00]]> : tensor<3x5xf16>
    return %0 : tensor<3x5xf16>
  }
}

