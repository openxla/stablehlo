// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf16>
    %1 = call @expected() : () -> tensor<3x5xf16>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xf16>, tensor<3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf16> {
    %0 = stablehlo.constant dense<[[-3.683590e+00, -2.134770e+00, -3.003910e+00, -6.259770e-01, -1.418460e-01, -4.343260e-01], [-5.519530e+00, -1.516600e+00, -4.558110e-01, 5.253910e-01, -1.229490e+00, -4.968260e-01], [1.255860e+00, -3.035160e+00, -7.197270e-01, 1.397460e+00, -3.291020e+00, 3.605470e+00], [-2.716800e+00, 2.200930e-01, -2.236330e+00, 5.351560e+00, 7.387690e-01, -2.459720e-01]]> : tensor<4x6xf16>
    return %0 : tensor<4x6xf16>
  }
  func.func private @expected() -> tensor<3x5xf16> {
    %0 = stablehlo.constant dense<[[1.316250e+02, 8.867180e+00, -9.008780e-01, -1.147460e-01, 7.525630e-02], [-6.378130e+01, 3.019530e+00, 4.816890e-01, 5.941400e+00, -1.449220e+01], [4.558590e+00, -2.150390e+00, 2.407810e+01, -3.637500e+01, 4.312500e+00]]> : tensor<3x5xf16>
    return %0 : tensor<3x5xf16>
  }
}

