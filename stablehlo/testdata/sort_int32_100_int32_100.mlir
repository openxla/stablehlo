// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<100xi32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<100xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<100xi32>, tensor<100xi32>)
    %1:2 = call @expected() : () -> (tensor<100xi32>, tensor<100xi32>)
    %2:2 = "stablehlo.sort"(%0#0, %0#1) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
      %3 = stablehlo.compare  LT, %arg2, %arg3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4 = stablehlo.compare  LT, %arg0, %arg1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %5 = stablehlo.compare  EQ, %arg0, %arg1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %6 = stablehlo.and %5, %3 : tensor<i1>
      %7 = stablehlo.or %4, %6 : tensor<i1>
      stablehlo.return %7 : tensor<i1>
    }) : (tensor<100xi32>, tensor<100xi32>) -> (tensor<100xi32>, tensor<100xi32>)
    stablehlo.custom_call @check.expect_eq(%2#0, %1#0) {has_side_effect = true} : (tensor<100xi32>, tensor<100xi32>) -> ()
    stablehlo.custom_call @check.expect_eq(%2#1, %1#1) {has_side_effect = true} : (tensor<100xi32>, tensor<100xi32>) -> ()
    return %2#0, %2#1 : tensor<100xi32>, tensor<100xi32>
  }
  func.func private @inputs() -> (tensor<100xi32> {mhlo.layout_mode = "default"}, tensor<100xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]> : tensor<100xi32>
    %c_0 = stablehlo.constant dense<[-1, 0, 0, 3, -1, -7, -5, -2, 0, -3, -2, -2, -1, 2, -2, 2, -2, 0, 0, 0, -2, 4, 2, 0, -3, -1, -1, 0, 0, -2, -2, 1, 2, -1, 1, -4, 0, -3, 0, -1, 0, -2, -2, -5, 0, 0, -5, 3, 0, 0, 3, 2, -2, -3, -4, -1, 2, -1, -5, 1, -4, 0, -3, 0, -2, 2, 1, 3, 5, 0, -1, -4, 0, 3, -1, 1, 0, 2, 3, -1, 2, 0, 0, -4, 3, 0, -1, 0, 6, 2, 1, 0, -4, -1, 0, 2, -3, 0, 3, -1]> : tensor<100xi32>
    return %c, %c_0 : tensor<100xi32>, tensor<100xi32>
  }
  func.func private @expected() -> (tensor<100xi32> {mhlo.layout_mode = "default"}, tensor<100xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<100xi32>
    %c_0 = stablehlo.constant dense<[-5, -5, -4, -4, -4, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 5, -7, -5, -5, -4, -4, -4, -3, -3, -3, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 6]> : tensor<100xi32>
    return %c, %c_0 : tensor<100xi32>, tensor<100xi32>
  }
}
