// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf16>
    %1 = call @expected() : () -> tensor<3x5xf16>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f16>) -> tensor<f16>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %6 : tensor<f16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xf16>, tensor<3x5xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf16> {
    %0 = stablehlo.constant dense<[[3.391110e-01, 4.191410e+00, -8.081050e-01, 5.522460e-01, -3.920900e-01, 3.277340e+00], [8.417960e-01, -3.046880e+00, 2.953130e+00, -6.082030e+00, 2.003910e+00, -7.470700e-01], [4.052730e-01, 3.464840e+00, 8.078130e+00, 9.877920e-01, 2.521480e+00, 3.408200e+00], [-7.578130e-01, -3.050780e+00, -3.521480e+00, 4.133300e-01, 3.757810e+00, -3.427730e+00]]> : tensor<4x6xf16>
    return %0 : tensor<4x6xf16>
  }
  func.func private @expected() -> tensor<3x5xf16> {
    %0 = stablehlo.constant dense<[[2.328130e+00, 3.289060e+00, -3.384770e+00, -3.917970e+00, 4.144530e+00], [1.665040e+00, 1.145310e+01, 5.937500e+00, -5.683590e-01, 7.187500e+00], [6.250000e-02, 4.976560e+00, 5.953130e+00, 7.679680e+00, 6.257810e+00]]> : tensor<3x5xf16>
    return %0 : tensor<3x5xf16>
  }
}

