// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xui16> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xui16>
    %1 = call @expected() : () -> tensor<5x7xui16>
    %2 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %3 = stablehlo.compare  LT, %arg0, %arg1,  UNSIGNED : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) : (tensor<5x7xui16>) -> tensor<5x7xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x7xui16>, tensor<5x7xui16>) -> ()
    return %2 : tensor<5x7xui16>
  }
  func.func private @inputs() -> (tensor<5x7xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 0, 0, 3, 0, 0, 3], [5, 1, 0, 2, 0, 1, 5], [0, 3, 1, 1, 2, 2, 1], [0, 1, 1, 0, 0, 1, 3], [3, 0, 4, 0, 1, 0, 1]]> : tensor<5x7xui16>
    return %c : tensor<5x7xui16>
  }
  func.func private @expected() -> (tensor<5x7xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [2, 1, 1, 1, 0, 1, 3], [3, 1, 1, 2, 1, 1, 3], [5, 3, 4, 3, 2, 2, 5]]> : tensor<5x7xui16>
    return %c : tensor<5x7xui16>
  }
}
