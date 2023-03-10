// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf16>
    %1 = call @expected() : () -> tensor<3x5xf16>
    %2 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f16>) -> tensor<f16>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %6 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %6 : tensor<f16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xf16>, tensor<3x5xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf16> {
    %0 = stablehlo.constant dense<[[-1.719970e-01, 4.230470e+00, -2.193360e+00, -6.246090e+00, 3.052730e+00, 5.601560e+00], [2.791020e+00, 9.882810e-01, -2.203130e+00, -3.742190e+00, 4.222660e+00, -4.438480e-01], [2.117920e-01, -5.277340e+00, 1.456050e+00, 3.759770e+00, 6.440430e-01, 2.861020e-02], [8.227530e-01, -3.175780e+00, -3.736330e+00, 8.154300e-01, 2.515630e+00, -2.839840e+00]]> : tensor<4x6xf16>
    return %0 : tensor<4x6xf16>
  }
  func.func private @expected() -> tensor<3x5xf16> {
    %0 = stablehlo.constant dense<[[4.230470e+00, 4.230470e+00, -2.193360e+00, 4.222660e+00, 5.601560e+00], [2.791020e+00, 1.456050e+00, 3.759770e+00, 4.222660e+00, 4.222660e+00], [8.227530e-01, 1.456050e+00, 3.759770e+00, 3.759770e+00, 2.515630e+00]]> : tensor<3x5xf16>
    return %0 : tensor<3x5xf16>
  }
}

