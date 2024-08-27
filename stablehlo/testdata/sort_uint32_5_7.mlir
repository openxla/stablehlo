// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xui32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xui32>
    %1 = call @expected() : () -> tensor<5x7xui32>
    %2 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %3 = stablehlo.compare  LT, %arg0, %arg1,  UNSIGNED : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) : (tensor<5x7xui32>) -> tensor<5x7xui32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x7xui32>, tensor<5x7xui32>) -> ()
    return %2 : tensor<5x7xui32>
  }
  func.func private @inputs() -> (tensor<5x7xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 5, 6, 0, 0, 0, 1], [4, 1, 5, 0, 0, 2, 2], [1, 4, 2, 0, 5, 0, 5], [3, 4, 3, 0, 4, 4, 1], [0, 3, 3, 2, 2, 1, 0]]> : tensor<5x7xui32>
    return %c : tensor<5x7xui32>
  }
  func.func private @expected() -> (tensor<5x7xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1, 2, 0, 0, 0, 0], [1, 3, 3, 0, 0, 0, 1], [1, 4, 3, 0, 2, 1, 1], [3, 4, 5, 0, 4, 2, 2], [4, 5, 6, 2, 5, 4, 5]]> : tensor<5x7xui32>
    return %c : tensor<5x7xui32>
  }
}
