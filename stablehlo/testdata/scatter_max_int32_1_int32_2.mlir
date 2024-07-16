// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<1xi32>, tensor<2xi32>)
    %1 = call @expected() : () -> tensor<1xi32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i32>
      stablehlo.return %3 : tensor<i32>
    }) : (tensor<1xi32>, tensor<2x1xi64>, tensor<2xi32>) -> tensor<1xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1xi32>, tensor<1xi32>) -> ()
    return %2 : tensor<1xi32>
  }
  func.func private @inputs() -> (tensor<1xi32> {mhlo.layout_mode = "default"}, tensor<2xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<-3> : tensor<1xi32>
    %c_0 = stablehlo.constant dense<[0, 2]> : tensor<2xi32>
    return %c, %c_0 : tensor<1xi32>, tensor<2xi32>
  }
  func.func private @expected() -> (tensor<1xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<2> : tensor<1xi32>
    return %c : tensor<1xi32>
  }
}
