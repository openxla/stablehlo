// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4xi32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xi32>
    %1 = call @expected() : () -> tensor<4xi32>
    %c = stablehlo.constant dense<3> : tensor<i32>
    %2 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [1] : (tensor<4x6xi32>, tensor<i32>) -> tensor<4xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4xi32>, tensor<4xi32>) -> ()
    return %2 : tensor<4xi32>
  }
  func.func private @inputs() -> (tensor<4x6xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 3, 7, 2, -1, -7], [5, 0, 2, -4, -3, 1], [-1, 7, 0, -1, 3, -3], [-4, 2, -3, -7, 0, -1]]> : tensor<4x6xi32>
    %c_0 = stablehlo.constant dense<[[2, 3, 0, 0, 0, 4], [7, 2, -5, -1, 1, 2], [0, 0, 0, 0, 4, -6], [-1, 0, -7, 2, 3, 0]]> : tensor<4x6xi32>
    return %c : tensor<4x6xi32>
  }
  func.func private @expected() -> (tensor<4xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[9, 4, 8, -10]> : tensor<4xi32>
    return %c : tensor<4xi32>
  }
}
