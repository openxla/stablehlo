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
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<4x2x3xf32>, tensor<2xi64>, tensor<2xf32>) -> tensor<4x2x3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> ()
    return %2 : tensor<4x2x3xf32>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32> {mhlo.layout_mode = "default"}, tensor<2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[1.38732839, 2.89946914, -2.876290e-01], [6.79478884, 1.76005256, 0.305416048]], [[1.63096201, 4.79242373, -1.44652247], [-0.525784791, -0.114975639, -1.21797776]], [[2.14421749, -0.641117454, -1.57080615], [1.77755022, -4.11711407, 3.59519815]], [[-2.79984736, -1.47077167, 3.22873116], [-1.34054029, -6.97518253, 0.675602198]]]> : tensor<4x2x3xf32>
    %cst_0 = stablehlo.constant dense<[0.105449297, 3.63069248]> : tensor<2xf32>
    return %cst, %cst_0 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> (tensor<4x2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[1.38732839, 2.89946914, -2.876290e-01], [6.79478884, 1.76005256, 0.305416048]], [[1.63096201, 4.79242373, -1.44652247], [-0.525784791, -0.114975639, -1.21797776]], [[2.14421749, -0.641117454, -1.57080615], [1.77755022, -4.11711407, 3.59519815]], [[-2.79984736, -1.47077167, 3.22873116], [-1.34054029, -6.97518253, 3.63069248]]]> : tensor<4x2x3xf32>
    return %cst : tensor<4x2x3xf32>
  }
}
