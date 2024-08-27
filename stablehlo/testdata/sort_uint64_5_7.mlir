// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xui64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xui64>
    %1 = call @expected() : () -> tensor<5x7xui64>
    %2 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<ui64>, %arg1: tensor<ui64>):
      %3 = stablehlo.compare  LT, %arg0, %arg1,  UNSIGNED : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) : (tensor<5x7xui64>) -> tensor<5x7xui64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x7xui64>, tensor<5x7xui64>) -> ()
    return %2 : tensor<5x7xui64>
  }
  func.func private @inputs() -> (tensor<5x7xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 1, 1, 1, 0, 1, 1], [0, 2, 0, 1, 2, 4, 5], [0, 1, 2, 0, 5, 1, 2], [1, 4, 2, 1, 2, 0, 4], [0, 2, 1, 2, 3, 3, 3]]> : tensor<5x7xui64>
    return %c : tensor<5x7xui64>
  }
  func.func private @expected() -> (tensor<5x7xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1, 0, 0, 0, 0, 1], [0, 1, 1, 1, 2, 1, 2], [0, 2, 1, 1, 2, 1, 3], [1, 2, 2, 1, 3, 3, 4], [3, 4, 2, 2, 5, 4, 5]]> : tensor<5x7xui64>
    return %c : tensor<5x7xui64>
  }
}
