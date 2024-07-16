// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui64>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui64>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    stablehlo.custom_call @check.expect_almost_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    return %4 : tensor<4x6xf32>
  }
  func.func private @inputs() -> (tensor<4x3xui64> {mhlo.layout_mode = "default"}, tensor<3x6xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 2, 0], [0, 1, 1], [1, 1, 1], [1, 3, 2]]> : tensor<4x3xui64>
    %cst = stablehlo.constant dense<[[2.69352365, -1.75565517, 3.64608979, -4.61772156, 0.969550549, 3.7079103], [-0.878219723, -1.76270247, 3.62539315, 3.14396954, -2.84121752, 1.35875285], [1.46522605, 7.14490747, 4.94334316, -4.47644567, -0.379561633, -3.70317078]]> : tensor<3x6xf32>
    return %c, %cst : tensor<4x3xui64>, tensor<3x6xf32>
  }
  func.func private @expected() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.937084197, -5.281060e+00, 10.8968763, 1.67021751, -4.71288443, 6.42541599], [0.58700633, 5.38220501, 8.56873607, -1.33247614, -3.22077918, -2.34441805], [3.280530e+00, 3.62654972, 12.2148266, -5.9501977, -2.25122857, 1.36349249], [2.98931646, 7.24605227, 24.4089546, -4.1387043, -8.31322479, 0.377827168]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
}
