// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x3x5xf32>, tensor<2x4x6xf32>)
    %1 = call @expected() : () -> tensor<2x4x6xf32>
    %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %2 = stablehlo.pad %0#1, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %cst_0) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.compare  LE, %arg0, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<2x4x6xf32>, tensor<1x3x5xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> ()
    return %4 : tensor<2x4x6xf32>
  }
  func.func private @inputs() -> (tensor<1x3x5xf32> {mhlo.layout_mode = "default"}, tensor<2x4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[3.68262458, 3.82101059, -2.2947979, 2.8329246, -1.15248716], [3.55029321, 1.37898874, 3.77211261, 2.91995788, 4.76481152], [-0.391007394, 1.91863632, -0.352135569, -2.31635451, 1.65467286]]]> : tensor<1x3x5xf32>
    %cst_0 = stablehlo.constant dense<[[[-3.01613498, -1.59692681, 1.84716642, 3.11134791, 1.66431081, 0.179143101], [-7.45665121, -1.83812463, 4.40967798, -1.26709688, 0.945837199, -4.26534128], [-6.25084066, 0.386573404, 3.05768943, 5.35562325, 1.75902784, 6.14879847], [-2.54304743, -0.416835576, -5.58269119, 6.80337095, 5.6900878, -5.19403219]], [[-3.99047351, -0.0503688157, -2.61801505, -1.74413991, 3.63066554, -0.434025705], [2.22170758, -3.62837696, 0.0174580626, 3.51287198, 5.15925407, 0.14256601], [0.16070427, -1.49531329, -2.80724978, -3.98462272, -1.87461674, 2.1211102], [0.858028948, -0.988228261, -3.62104201, -0.313145071, -1.7683264, 2.46107268]]]> : tensor<2x4x6xf32>
    return %cst, %cst_0 : tensor<1x3x5xf32>, tensor<2x4x6xf32>
  }
  func.func private @expected() -> (tensor<2x4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [7.23291778, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.61232424], [-0.391007394, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.56650078, 0.000000e+00, 0.000000e+00, 1.65467286]], [[0.000000e+00, 0.000000e+00, -2.2947979, 2.8329246, 0.000000e+00, 0.000000e+00], [0.000000e+00, 5.19999933, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 4.37571621, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>
    return %cst : tensor<2x4x6xf32>
  }
}
