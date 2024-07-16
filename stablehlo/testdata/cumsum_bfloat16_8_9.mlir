// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xbf16>
    %1 = call @expected() : () -> tensor<8x9xbf16>
    %2 = call @cumsum(%0) : (tensor<8x9xbf16>) -> tensor<8x9xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<8x9xbf16>, tensor<8x9xbf16>) -> ()
    return %2 : tensor<8x9xbf16>
  }
  func.func private @inputs() -> (tensor<8x9xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.437500e+00, 1.367190e+00, 7.460930e-01, -4.093750e+00, -2.437500e+00, -8.312500e+00, -1.750000e+00, -3.015630e+00, 3.171880e+00], [-6.796880e-01, -5.781250e+00, -3.093750e+00, 6.000000e+00, 1.265630e+00, -3.250000e+00, -1.865230e-01, -2.218750e+00, 6.884770e-02], [1.203130e+00, -7.382810e-01, -2.625000e+00, -2.765630e+00, -3.281250e+00, 2.625000e+00, -5.468750e+00, -6.062500e+00, 6.093750e+00], [-6.367190e-01, 3.203130e+00, 5.585940e-01, -2.750000e+00, 2.281250e+00, 1.037500e+01, -2.093750e+00, -3.218750e+00, 6.468750e+00], [-2.468750e+00, -3.925780e-01, 3.906250e+00, -2.703130e+00, 3.218750e+00, 3.078130e+00, 3.906250e+00, -5.187500e+00, -1.812500e+00], [8.375000e+00, -1.312500e+00, 1.867190e+00, 4.187500e+00, -9.296870e-01, 2.203130e+00, -5.039060e-01, 6.750000e+00, 2.781250e+00], [1.125000e+00, -3.171880e+00, -3.546880e+00, 5.687500e+00, -1.376950e-01, 1.218750e+00, 4.199220e-01, 9.960930e-01, 3.078130e+00], [-2.910160e-01, -1.148440e+00, 3.187500e+00, -3.109380e+00, 1.085940e+00, 5.156250e-01, -5.312500e+00, 7.250000e+00, -2.421880e+00]]> : tensor<8x9xbf16>
    return %cst : tensor<8x9xbf16>
  }
  func.func private @expected() -> (tensor<8x9xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.437500e+00, 1.367190e+00, 7.460930e-01, -4.093750e+00, -2.437500e+00, -8.312500e+00, -1.750000e+00, -3.015630e+00, 3.171880e+00], [3.750000e+00, -4.406250e+00, -2.343750e+00, 1.906250e+00, -1.171880e+00, -1.156250e+01, -1.937500e+00, -5.250000e+00, 3.234380e+00], [4.937500e+00, -5.156250e+00, -4.968750e+00, -8.593750e-01, -4.437500e+00, -8.937500e+00, -7.406250e+00, -1.131250e+01, 9.312500e+00], [4.312500e+00, -1.953130e+00, -4.406250e+00, -3.609380e+00, -2.156250e+00, 1.437500e+00, -9.500000e+00, -1.450000e+01, 1.575000e+01], [1.843750e+00, -2.343750e+00, -5.000000e-01, -6.312500e+00, 1.062500e+00, 4.500000e+00, -5.593750e+00, -1.975000e+01, 1.393750e+01], [1.025000e+01, -3.656250e+00, 1.367190e+00, -2.125000e+00, 1.328130e-01, 6.687500e+00, -6.093750e+00, -1.300000e+01, 1.675000e+01], [1.137500e+01, -6.812500e+00, -2.187500e+00, 3.562500e+00, -4.882810e-03, 7.906250e+00, -5.687500e+00, -1.200000e+01, 1.987500e+01], [1.106250e+01, -7.968750e+00, 1.000000e+00, 4.531250e-01, 1.078130e+00, 8.437500e+00, -1.100000e+01, -4.750000e+00, 1.750000e+01]]> : tensor<8x9xbf16>
    return %cst : tensor<8x9xbf16>
  }
  func.func private @cumsum(%arg0: tensor<8x9xbf16>) -> tensor<8x9xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<bf16>
      stablehlo.return %2 : tensor<bf16>
    }) : (tensor<8x9xbf16>, tensor<bf16>) -> tensor<8x9xbf16>
    return %1 : tensor<8x9xbf16>
  }
}
