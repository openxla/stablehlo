// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3xf32>, tensor<2xf32>)
    %1 = call @expected() : () -> tensor<4x2x3xf32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<4x2x3xf32>, tensor<2xi64>, tensor<2xf32>) -> tensor<4x2x3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> ()
    return %2 : tensor<4x2x3xf32>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32> {mhlo.layout_mode = "default"}, tensor<2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[4.49575233, 2.57913351, 3.17360067], [2.86800289, 0.764879227, 3.50482368]], [[-0.787247478, -2.3107903, -0.8217265], [-5.53171682, 2.28917766, -5.91439295]], [[-2.06035209, 1.88200915, -1.49790323], [7.83903265, -1.51898813, -1.68348587]], [[2.52399182, -5.52001381, 1.54348922], [-2.56950092, -4.03907061, -1.26045203]]]> : tensor<4x2x3xf32>
    %cst_0 = stablehlo.constant dense<[1.93206847, 1.86264634]> : tensor<2xf32>
    return %cst, %cst_0 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> (tensor<4x2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[4.49575233, 2.57913351, 3.17360067], [2.86800289, 0.764879227, 3.50482368]], [[-0.787247478, -2.3107903, -0.8217265], [-5.53171682, 2.28917766, -5.91439295]], [[-2.06035209, 1.88200915, -1.49790323], [7.83903265, -1.51898813, -1.68348587]], [[2.52399182, -5.52001381, 2.98212695], [-2.56950092, -4.03907061, -2.34777641]]]> : tensor<4x2x3xf32>
    return %cst : tensor<4x2x3xf32>
  }
}
