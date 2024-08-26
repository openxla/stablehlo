// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xi1> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<6xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xi1>, tensor<4x6xi32>)
    %1:2 = call @expected() : () -> (tensor<6xi1>, tensor<6xi32>)
    %c = stablehlo.constant dense<true> : tensor<i1>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %2:2 = stablehlo.reduce(%0#0 init: %c), (%0#1 init: %c_0) across dimensions = [0] : (tensor<4x6xi1>, tensor<4x6xi32>, tensor<i1>, tensor<i32>) -> (tensor<6xi1>, tensor<6xi32>)
     reducer(%arg0: tensor<i1>, %arg2: tensor<i1>) (%arg1: tensor<i32>, %arg3: tensor<i32>)  {
      %3 = stablehlo.maximum %arg0, %arg2 : tensor<i1>
      %4 = stablehlo.minimum %arg1, %arg3 : tensor<i32>
      stablehlo.return %3, %4 : tensor<i1>, tensor<i32>
    }
    stablehlo.custom_call @check.expect_eq(%2#0, %1#0) {has_side_effect = true} : (tensor<6xi1>, tensor<6xi1>) -> ()
    stablehlo.custom_call @check.expect_eq(%2#1, %1#1) {has_side_effect = true} : (tensor<6xi32>, tensor<6xi32>) -> ()
    return %2#0, %2#1 : tensor<6xi1>, tensor<6xi32>
  }
  func.func private @inputs() -> (tensor<4x6xi1> {mhlo.layout_mode = "default"}, tensor<4x6xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<4x6xi1>
    %c_0 = stablehlo.constant dense<[[-3, 0, -2, 0, 0, -2], [1, 10, 5, 0, 4, 1], [0, 3, 0, 3, 0, -1], [2, 8, 0, -2, 3, -2]]> : tensor<4x6xi32>
    return %c, %c_0 : tensor<4x6xi1>, tensor<4x6xi32>
  }
  func.func private @expected() -> (tensor<6xi1> {mhlo.layout_mode = "default"}, tensor<6xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<6xi1>
    %c_0 = stablehlo.constant dense<[-3, 0, -2, -2, 0, -2]> : tensor<6xi32>
    return %c, %c_0 : tensor<6xi1>, tensor<6xi32>
  }
}
