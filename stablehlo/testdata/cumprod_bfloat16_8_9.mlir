// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xbf16>
    %1 = call @expected() : () -> tensor<8x9xbf16>
    %2 = call @cumprod(%0) : (tensor<8x9xbf16>) -> tensor<8x9xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<8x9xbf16>, tensor<8x9xbf16>) -> ()
    return %2 : tensor<8x9xbf16>
  }
  func.func private @inputs() -> (tensor<8x9xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.929690e+00, -4.625000e+00, 1.460940e+00, 1.890630e+00, -1.546880e+00, 2.250000e+00, -1.984380e+00, 1.179690e+00, 4.570310e-01], [-2.437500e+00, 1.718750e+00, 4.500000e+00, -1.890630e+00, -5.078130e-01, -1.669920e-01, 7.421880e-01, -7.019040e-03, 1.914060e+00], [-4.531250e+00, -1.867190e+00, -2.109380e+00, -2.421880e+00, -4.906250e+00, -2.140630e+00, 2.812500e+00, 2.156250e+00, 1.757810e+00], [1.625000e+00, 1.734380e+00, -1.085940e+00, 3.796880e+00, 8.750000e-01, -1.195310e+00, 1.570310e+00, -4.968750e+00, 1.234380e+00], [2.359380e+00, 1.382810e+00, -2.314450e-01, 7.382810e-01, -2.468750e+00, 4.843750e+00, 1.734380e+00, -2.187500e+00, 3.640630e+00], [-1.593750e+00, -3.417970e-01, -3.625000e+00, -1.187500e+00, -1.765630e+00, 2.968750e+00, -6.375000e+00, -1.062500e+00, -1.984380e+00], [-2.125000e+00, -9.492180e-01, 2.671880e+00, 1.656250e+00, -1.343750e+00, 2.015630e+00, 4.187500e+00, 3.031250e+00, 6.093750e-01], [3.390630e+00, 9.687500e+00, 2.453130e+00, -2.750000e+00, 2.953130e+00, 8.554680e-01, 5.937500e-01, 3.710940e-01, -9.570310e-01]]> : tensor<8x9xbf16>
    return %cst : tensor<8x9xbf16>
  }
  func.func private @expected() -> (tensor<8x9xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.929690e+00, -4.625000e+00, 1.460940e+00, 1.890630e+00, -1.546880e+00, 2.250000e+00, -1.984380e+00, 1.179690e+00, 4.570310e-01], [4.718750e+00, -7.937500e+00, 6.562500e+00, -3.578130e+00, 7.851560e-01, -3.750000e-01, -1.476560e+00, -8.300780e-03, 8.750000e-01], [-2.137500e+01, 1.481250e+01, -1.381250e+01, 8.687500e+00, -3.859380e+00, 8.046880e-01, -4.156250e+00, -1.794430e-02, 1.539060e+00], [-3.475000e+01, 2.575000e+01, 1.500000e+01, 3.300000e+01, -3.375000e+00, -9.609370e-01, -6.531250e+00, 8.935540e-02, 1.898440e+00], [-8.200000e+01, 3.550000e+01, -3.468750e+00, 2.437500e+01, 8.312500e+00, -4.656250e+00, -1.131250e+01, -1.953130e-01, 6.906250e+00], [1.310000e+02, -1.212500e+01, 1.256250e+01, -2.900000e+01, -1.468750e+01, -1.381250e+01, 7.200000e+01, 2.070310e-01, -1.368750e+01], [-2.780000e+02, 1.150000e+01, 3.350000e+01, -4.800000e+01, 1.975000e+01, -2.787500e+01, 3.020000e+02, 6.289060e-01, -8.312500e+00], [-9.440000e+02, 1.115000e+02, 8.200000e+01, 1.320000e+02, 5.825000e+01, -2.387500e+01, 1.790000e+02, 2.333980e-01, 7.968750e+00]]> : tensor<8x9xbf16>
    return %cst : tensor<8x9xbf16>
  }
  func.func private @cumprod(%arg0: tensor<8x9xbf16>) -> tensor<8x9xbf16> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.multiply %arg1, %arg2 : tensor<bf16>
      stablehlo.return %1 : tensor<bf16>
    }) : (tensor<8x9xbf16>, tensor<bf16>) -> tensor<8x9xbf16>
    return %0 : tensor<8x9xbf16>
  }
}
