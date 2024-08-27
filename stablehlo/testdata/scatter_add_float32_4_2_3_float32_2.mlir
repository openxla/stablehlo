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
      %3 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<4x2x3xf32>, tensor<2xi64>, tensor<2xf32>) -> tensor<4x2x3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> ()
    return %2 : tensor<4x2x3xf32>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32> {mhlo.layout_mode = "default"}, tensor<2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[4.823510e+00, 0.423392832, 2.67835927], [-2.08515167, -3.6946454, 2.47799826]], [[-2.64480543, -0.643004477, 5.69479227], [-3.7926693, -1.77346087, -1.33025837]], [[-1.96124375, -0.377599716, -3.10537505], [-3.95107126, -1.77462971, 2.59804296]], [[-6.70349025, -0.822746276, 2.51327872], [-1.05100095, 3.29730201, 5.58889532]]]> : tensor<4x2x3xf32>
    %cst_0 = stablehlo.constant dense<[5.25133562, -1.03823602]> : tensor<2xf32>
    return %cst, %cst_0 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> (tensor<4x2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[4.823510e+00, 0.423392832, 2.67835927], [-2.08515167, -3.6946454, 2.47799826]], [[-2.64480543, -0.643004477, 5.69479227], [-3.7926693, -1.77346087, -1.33025837]], [[-1.96124375, -0.377599716, -3.10537505], [-3.95107126, -1.77462971, 2.59804296]], [[-6.70349025, -0.822746276, 7.76461411], [-1.05100095, 3.29730201, 4.55065918]]]> : tensor<4x2x3xf32>
    return %cst : tensor<4x2x3xf32>
  }
}
