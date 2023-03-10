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
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xf16>, tensor<3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf16> {
    %0 = stablehlo.constant dense<[[7.983390e-01, 6.962890e-01, 3.402340e+00, -3.615720e-01, 2.423830e+00, -2.826170e+00], [-4.851560e+00, 1.663090e+00, -2.668760e-02, 2.546880e+00, 8.701170e-01, 2.376950e+00], [9.770500e-01, -1.735350e+00, 1.783200e+00, 2.802730e+00, 4.570310e+00, -4.035160e+00], [5.415040e-01, 6.967770e-01, -1.946290e+00, -1.960940e+00, 5.234380e+00, -3.752440e-01]]> : tensor<4x6xf16>
    return %0 : tensor<4x6xf16>
  }
  func.func private @expected() -> tensor<3x5xf16> {
    %0 = stablehlo.constant dense<[[-6.923830e-01, 6.734380e+00, 6.558590e+00, 6.480460e+00, 3.843750e+00], [-2.945310e+00, 2.683590e+00, 8.109380e+00, 1.178910e+01, 4.777340e+00], [1.479490e+00, -2.021480e-01, 1.679690e+00, 1.164840e+01, 6.394530e+00]]> : tensor<3x5xf16>
    return %0 : tensor<3x5xf16>
  }
}

