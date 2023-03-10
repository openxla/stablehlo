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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xf16>, tensor<3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf16> {
    %0 = stablehlo.constant dense<[[-2.677730e+00, 2.482420e+00, 2.773440e-01, -2.341800e+00, 5.285160e+00, -9.418940e-01], [8.978270e-02, 3.193360e-01, -1.239260e+00, 3.919920e+00, -1.218750e+00, -1.467770e+00], [1.302730e+00, -2.031250e+00, -1.469730e+00, 2.199220e+00, -7.963860e-01, -3.176270e-01], [-1.118160e+00, 1.302730e+00, -3.222660e+00, 2.321780e-01, 2.128910e+00, -3.205080e+00]]> : tensor<4x6xf16>
    return %0 : tensor<4x6xf16>
  }
  func.func private @expected() -> tensor<3x5xf16> {
    %0 = stablehlo.constant dense<[[2.482420e+00, 2.482420e+00, 3.919920e+00, 5.285160e+00, 5.285160e+00], [1.302730e+00, 1.000000e+00, 3.919920e+00, 3.919920e+00, 1.000000e+00], [1.302730e+00, 1.302730e+00, 2.199220e+00, 2.199220e+00, 2.128910e+00]]> : tensor<3x5xf16>
    return %0 : tensor<3x5xf16>
  }
}

