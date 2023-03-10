// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf16>
    %1 = call @expected() : () -> tensor<3x5xf16>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xf16>, tensor<3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf16> {
    %0 = stablehlo.constant dense<[[4.082030e+00, 1.345700e+00, 1.520510e+00, 4.206540e-01, -3.919920e+00, 2.216800e+00], [-2.736330e+00, 7.895500e-01, -2.047120e-01, -4.824220e+00, 2.492190e+00, -1.348630e+00], [5.640630e+00, -1.086910e+00, 6.902340e+00, -1.696780e-01, -8.813470e-01, -3.659670e-01], [-3.802730e+00, 2.300780e+00, -3.930660e-01, -1.002930e+00, 2.683590e+00, 1.181640e+00]]> : tensor<4x6xf16>
    return %0 : tensor<4x6xf16>
  }
  func.func private @expected() -> tensor<3x5xf16> {
    %0 = stablehlo.constant dense<[[-1.186720e+01, -3.308110e-01, 6.318360e-01, 1.982810e+01, 2.920310e+01], [1.325000e+01, 1.212890e+00, -1.156250e+00, -1.798830e+00, -1.083980e+00], [5.368750e+01, 6.785150e+00, -4.616700e-01, -4.025880e-01, 1.023440e+00]]> : tensor<3x5xf16>
    return %0 : tensor<3x5xf16>
  }
}

