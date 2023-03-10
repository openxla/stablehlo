// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xbf16>
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %2 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %6 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %6 : tensor<bf16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xbf16> {
    %0 = stablehlo.constant dense<[[3.250000e+00, -2.468750e+00, 3.796880e+00, 2.484380e+00, -2.515630e+00, -4.218750e+00], [-3.625000e+00, 1.171880e+00, 5.156250e+00, -1.921880e+00, -3.421880e+00, 7.156250e+00], [1.453130e+00, 1.179690e+00, -4.726560e-01, 3.937500e+00, 2.062500e+00, -7.421880e-01], [-7.070310e-01, 4.687500e+00, 9.062500e-01, 5.718750e+00, 2.125000e+00, -1.953130e+00]]> : tensor<4x6xbf16>
    return %0 : tensor<4x6xbf16>
  }
  func.func private @expected() -> tensor<3x5xbf16> {
    %0 = stablehlo.constant dense<[[3.250000e+00, 5.156250e+00, 5.156250e+00, 2.484380e+00, 7.156250e+00], [1.453130e+00, 5.156250e+00, 5.156250e+00, 3.937500e+00, 7.156250e+00], [4.687500e+00, 4.687500e+00, 5.718750e+00, 5.718750e+00, 2.125000e+00]]> : tensor<3x5xbf16>
    return %0 : tensor<3x5xbf16>
  }
}

