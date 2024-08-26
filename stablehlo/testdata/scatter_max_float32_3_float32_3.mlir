// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1], [0], [1]]> : tensor<3x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3xf32>, tensor<3xf32>)
    %1 = call @expected() : () -> tensor<3xf32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<3xf32>, tensor<3x1xi64>, tensor<3xf32>) -> tensor<3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xf32>, tensor<3xf32>) -> ()
    return %2 : tensor<3xf32>
  }
  func.func private @inputs() -> (tensor<3xf32> {mhlo.layout_mode = "default"}, tensor<3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[2.29838133, -3.78292799, 0.110196404]> : tensor<3xf32>
    %cst_0 = stablehlo.constant dense<[-2.69816899, -4.59641647, 1.902650e+00]> : tensor<3xf32>
    return %cst, %cst_0 : tensor<3xf32>, tensor<3xf32>
  }
  func.func private @expected() -> (tensor<3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[2.29838133, 1.902650e+00, 0.110196404]> : tensor<3xf32>
    return %cst : tensor<3xf32>
  }
}
