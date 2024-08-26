// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3x4xf32>, tensor<2x4x6xf32>)
    %1 = call @expected() : () -> tensor<2x4x6xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.pad %0#1, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %cst_0) <{window_dimensions = array<i64: 1, 2, 3>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<2x4x6xf32>, tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> ()
    return %4 : tensor<2x4x6xf32>
  }
  func.func private @inputs() -> (tensor<2x3x4xf32> {mhlo.layout_mode = "default"}, tensor<2x4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-0.141975358, 0.899714291, 2.56063509, 4.19124699], [-3.65281343, -1.86750543, -4.47070408, -2.39011383], [-6.062790e+00, 2.59964061, -0.644573092, -2.90732551]], [[-0.199792087, 6.34653139, 6.94955348, 4.45443439], [2.708630e+00, 1.3340112, 4.9623251, -1.93976092], [0.264342725, 3.3276751, 2.26464963, 1.62383819]]]> : tensor<2x3x4xf32>
    %cst_0 = stablehlo.constant dense<[[[-4.83879852, 0.13970837, 1.20698881, 0.763655662, -1.41355038, -0.808346152], [4.13756514, -2.28614664, -2.28975868, 1.78466904, 0.397792876, 0.802912414], [1.32220972, -3.08647823, 1.04847288, -0.915023326, -0.125590891, 1.63029754], [3.96157026, 5.32061863, -1.98978925, -5.22456074, 0.259640813, -3.288920e+00]], [[-0.2864438, -0.0922905281, -1.75957131, -2.51840854, 2.27084374, -1.36041808], [0.440647334, -1.8337611, -2.76243925, 1.22662556, -5.52954626, 1.53385413], [-0.209350288, -1.79192138, -3.08719182, 1.85777771, 1.66460097, 3.35604095], [0.884390711, -3.70861816, 0.817521929, -0.586911082, 0.451280475, -2.13313031]]]> : tensor<2x4x6xf32>
    return %cst, %cst_0 : tensor<2x3x4xf32>, tensor<2x4x6xf32>
  }
  func.func private @expected() -> (tensor<2x4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-3.79478884, 0.000000e+00, 0.000000e+00, -1.07672739, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, -0.644573092, 0.000000e+00, 0.000000e+00, -2.90732551], [0.000000e+00, -3.46314931, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 11.4039879, 0.000000e+00], [2.50883794, 0.000000e+00, 0.000000e+00, 6.34653139, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 11.8886604, 0.000000e+00, -0.315922737], [0.264342725, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>
    return %cst : tensor<2x4x6xf32>
  }
}
