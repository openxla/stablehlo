// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xbf16>
    %1 = call @expected() : () -> tensor<8x9xbf16>
    %2 = call @cummin(%0) : (tensor<8x9xbf16>) -> tensor<8x9xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<8x9xbf16>, tensor<8x9xbf16>) -> ()
    return %2 : tensor<8x9xbf16>
  }
  func.func private @inputs() -> (tensor<8x9xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.757810e+00, -3.359380e+00, 4.968750e+00, 5.195310e-01, -1.298830e-01, -3.242190e-01, 1.220700e-01, 4.000000e+00, 3.031250e+00], [-1.984380e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -4.941410e-01, 8.125000e-01, 2.937500e+00, 1.078130e+00], [3.031250e+00, 8.593750e-01, 2.171880e+00, -4.031250e+00, 2.453130e+00, 2.093750e+00, 2.140630e+00, -1.304690e+00, 3.125000e+00], [-3.484380e+00, -3.652340e-01, 3.593750e+00, 6.787110e-02, -2.328130e+00, -8.375000e+00, -2.046880e+00, 1.195310e+00, 7.375000e+00], [-2.187500e+00, -5.812500e+00, 2.171880e+00, 4.156250e+00, 1.875000e+00, 2.406250e+00, 6.812500e+00, -6.445310e-01, -1.734380e+00], [-4.250000e+00, -3.171880e+00, -3.906250e-02, 7.617180e-02, -5.039060e-01, 2.640630e+00, 7.343750e-01, 3.375000e+00, 3.437500e+00], [5.125000e+00, 3.234380e+00, -2.078130e+00, 3.476560e-01, -3.125000e+00, 1.820310e+00, -1.699220e-01, -6.031250e+00, -3.105470e-01], [1.164060e+00, -3.171880e+00, 4.550780e-01, 1.390630e+00, 3.109380e+00, 3.457030e-01, -7.773430e-01, -1.984380e+00, 1.515630e+00]]> : tensor<8x9xbf16>
    return %cst : tensor<8x9xbf16>
  }
  func.func private @expected() -> (tensor<8x9xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.757810e+00, -3.359380e+00, 4.968750e+00, 5.195310e-01, -1.298830e-01, -3.242190e-01, 1.220700e-01, 4.000000e+00, 3.031250e+00], [-1.984380e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -4.941410e-01, 1.220700e-01, 2.937500e+00, 1.078130e+00], [-1.984380e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -4.941410e-01, 1.220700e-01, -1.304690e+00, 1.078130e+00], [-3.484380e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -8.375000e+00, -2.046880e+00, -1.304690e+00, 1.078130e+00], [-3.484380e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -8.375000e+00, -2.046880e+00, -1.304690e+00, -1.734380e+00], [-4.250000e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -8.375000e+00, -2.046880e+00, -1.304690e+00, -1.734380e+00], [-4.250000e+00, -7.312500e+00, -2.078130e+00, -5.375000e+00, -3.234380e+00, -8.375000e+00, -2.046880e+00, -6.031250e+00, -1.734380e+00], [-4.250000e+00, -7.312500e+00, -2.078130e+00, -5.375000e+00, -3.234380e+00, -8.375000e+00, -2.046880e+00, -6.031250e+00, -1.734380e+00]]> : tensor<8x9xbf16>
    return %cst : tensor<8x9xbf16>
  }
  func.func private @cummin(%arg0: tensor<8x9xbf16>) -> tensor<8x9xbf16> {
    %cst = stablehlo.constant dense<0x7F80> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %2 = stablehlo.minimum %arg1, %arg2 : tensor<bf16>
      stablehlo.return %2 : tensor<bf16>
    }) : (tensor<8x9xbf16>, tensor<bf16>) -> tensor<8x9xbf16>
    return %1 : tensor<8x9xbf16>
  }
}
