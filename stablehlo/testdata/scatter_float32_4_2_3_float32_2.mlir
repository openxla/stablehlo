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
      stablehlo.return %arg1 : tensor<f32>
    }) : (tensor<4x2x3xf32>, tensor<2xi64>, tensor<2xf32>) -> tensor<4x2x3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> ()
    return %2 : tensor<4x2x3xf32>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32> {mhlo.layout_mode = "default"}, tensor<2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-3.85092044, -1.29653883, 5.04398203], [-1.77891552, 1.14304197, -0.242220432]], [[0.770432711, 0.497477651, 0.199629322], [1.07187033, 0.0254457798, 1.49424314]], [[-0.667058706, -6.894630e-01, -0.501317859], [0.405911714, -3.601150e+00, 2.04743695]], [[1.35089195, 0.783829689, 0.029527653], [2.21560669, -3.0994556, 0.691326737]]]> : tensor<4x2x3xf32>
    %cst_0 = stablehlo.constant dense<[0.18563509, -2.30085182]> : tensor<2xf32>
    return %cst, %cst_0 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> (tensor<4x2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-3.85092044, -1.29653883, 5.04398203], [-1.77891552, 1.14304197, -0.242220432]], [[0.770432711, 0.497477651, 0.199629322], [1.07187033, 0.0254457798, 1.49424314]], [[-0.667058706, -6.894630e-01, -0.501317859], [0.405911714, -3.601150e+00, 2.04743695]], [[1.35089195, 0.783829689, 0.18563509], [2.21560669, -3.0994556, -2.30085182]]]> : tensor<4x2x3xf32>
    return %cst : tensor<4x2x3xf32>
  }
}
