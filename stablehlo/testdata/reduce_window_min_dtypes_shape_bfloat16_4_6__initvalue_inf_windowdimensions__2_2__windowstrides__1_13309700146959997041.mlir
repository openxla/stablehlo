// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xbf16>
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %2 = stablehlo.constant dense<0x7F80> : tensor<bf16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %6 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %6 : tensor<bf16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xbf16> {
    %0 = stablehlo.constant dense<[[-1.484380e+00, 7.929680e-01, 1.226560e+00, 5.406250e+00, 5.625000e-01, -2.539060e-01], [1.203130e+00, 1.390630e+00, 1.312500e+00, -3.203130e+00, 1.359380e+00, -2.906250e+00], [-2.171880e+00, -4.718750e+00, 1.406250e+00, -3.234380e+00, -1.585940e+00, 4.093750e+00], [-3.093750e+00, 2.539060e-01, 3.671880e+00, 7.910150e-02, 2.765630e+00, 1.101560e+00]]> : tensor<4x6xbf16>
    return %0 : tensor<4x6xbf16>
  }
  func.func private @expected() -> tensor<3x5xbf16> {
    %0 = stablehlo.constant dense<[[-1.484380e+00, 7.929680e-01, -3.203130e+00, -3.203130e+00, -2.906250e+00], [-4.718750e+00, -4.718750e+00, -3.234380e+00, -3.234380e+00, -2.906250e+00], [-4.718750e+00, -4.718750e+00, -3.234380e+00, -3.234380e+00, -1.585940e+00]]> : tensor<3x5xbf16>
    return %0 : tensor<3x5xbf16>
  }
}

