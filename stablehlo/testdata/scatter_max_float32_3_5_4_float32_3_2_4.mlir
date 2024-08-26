// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5x4xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3x5x4xf32>, tensor<3x2x4xf32>)
    %1 = call @expected() : () -> tensor<3x5x4xf32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<3x5x4xf32>, tensor<2x1xi64>, tensor<3x2x4xf32>) -> tensor<3x5x4xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5x4xf32>, tensor<3x5x4xf32>) -> ()
    return %2 : tensor<3x5x4xf32>
  }
  func.func private @inputs() -> (tensor<3x5x4xf32> {mhlo.layout_mode = "default"}, tensor<3x2x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.795195758, 2.3342638, -8.81415653, 1.23357701], [-4.36560059, -0.638424456, -0.718834877, 4.14658785], [2.32594824, 0.143139109, -1.16659629, -0.544513524], [-2.59646702, 1.05930948, -2.3792398, 2.71689725], [0.953705549, 4.79323912, -1.43913305, 0.628115416]], [[2.95383573, -1.50061738, -1.05295682, -1.5377177], [-0.670464992, -3.36325407, 3.74503446, -0.628024041], [-4.02405691, 1.40161729, -2.95414472, 4.58786726], [-4.72267151, 1.75000799, -4.70211601, 1.46913254], [3.78041697, -0.302052796, -1.536770e+00, 3.1865685]], [[-5.1827426, -3.33841729, -2.0767405, -2.0219779], [-5.14500284, 0.72943753, 1.82685614, -1.79362619], [0.5942083, -4.41070271, -2.81045461, -1.6104219], [-4.34431887, 0.294988304, -7.3040452, -3.0086143], [-2.50675035, -1.44723225, -0.03330658, 8.06962394]]]> : tensor<3x5x4xf32>
    %cst_0 = stablehlo.constant dense<[[[0.577066958, -2.18397141, -2.8939085, -0.95165193], [1.09548867, -6.21386909, -3.05279326, -0.183454916]], [[-2.95009875, 1.10369265, 0.246283367, -1.64306867], [-2.18963671, -3.24388552, 5.34467602, 2.04179168]], [[-1.99170721, -3.33743191, 2.57429338, -0.718987762], [2.87324333, 1.26086748, -1.30653155, 4.29229879]]]> : tensor<3x2x4xf32>
    return %cst, %cst_0 : tensor<3x5x4xf32>, tensor<3x2x4xf32>
  }
  func.func private @expected() -> (tensor<3x5x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.795195758, 2.3342638, -8.81415653, 1.23357701], [1.09548867, -0.638424456, -0.718834877, 4.14658785], [2.32594824, 0.143139109, -1.16659629, -0.544513524], [-2.59646702, 1.05930948, -2.3792398, 2.71689725], [0.953705549, 4.79323912, -1.43913305, 0.628115416]], [[2.95383573, -1.50061738, -1.05295682, -1.5377177], [-0.670464992, 1.10369265, 5.34467602, 2.04179168], [-4.02405691, 1.40161729, -2.95414472, 4.58786726], [-4.72267151, 1.75000799, -4.70211601, 1.46913254], [3.78041697, -0.302052796, -1.536770e+00, 3.1865685]], [[-5.1827426, -3.33841729, -2.0767405, -2.0219779], [2.87324333, 1.26086748, 2.57429338, 4.29229879], [0.5942083, -4.41070271, -2.81045461, -1.6104219], [-4.34431887, 0.294988304, -7.3040452, -3.0086143], [-2.50675035, -1.44723225, -0.03330658, 8.06962394]]]> : tensor<3x5x4xf32>
    return %cst : tensor<3x5x4xf32>
  }
}
