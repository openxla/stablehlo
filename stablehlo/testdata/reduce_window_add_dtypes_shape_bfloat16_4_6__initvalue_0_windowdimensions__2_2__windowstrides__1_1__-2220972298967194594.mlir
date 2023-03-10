// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xbf16>
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %6 : tensor<bf16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xbf16> {
    %0 = stablehlo.constant dense<[[-5.937500e-01, -4.121090e-01, -7.906250e+00, 7.617180e-01, 6.375000e+00, -2.000000e+00], [2.046880e+00, 9.912100e-02, 1.046880e+00, -1.304690e+00, -3.796880e+00, 8.671870e-01], [3.625000e+00, -9.023430e-01, 1.782230e-02, 2.046880e+00, 3.890630e+00, -7.812500e-01], [6.015630e-01, -2.203130e+00, 2.171880e+00, 1.296880e+00, 2.468750e+00, 5.937500e-01]]> : tensor<4x6xbf16>
    return %0 : tensor<4x6xbf16>
  }
  func.func private @expected() -> tensor<3x5xbf16> {
    %0 = stablehlo.constant dense<[[1.140630e+00, -7.125000e+00, -7.437500e+00, 2.015630e+00, 1.445310e+00], [4.843750e+00, 2.636720e-01, 1.804690e+00, 8.437500e-01, 1.718750e-01], [1.109380e+00, -9.218750e-01, 5.562500e+00, 9.750000e+00, 6.156250e+00]]> : tensor<3x5xbf16>
    return %0 : tensor<3x5xbf16>
  }
}

