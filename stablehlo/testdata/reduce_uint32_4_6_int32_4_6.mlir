// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xui32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xui32>
    %1 = call @expected() : () -> tensor<6xui32>
    %c = stablehlo.constant dense<3> : tensor<ui32>
    %2 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<4x6xui32>, tensor<ui32>) -> tensor<6xui32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<6xui32>, tensor<6xui32>) -> ()
    return %2 : tensor<6xui32>
  }
  func.func private @inputs() -> (tensor<4x6xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, 3, 1, 2, 1, 1], [0, 1, 2, 0, 0, 3], [1, 4, 4, 2, 3, 4], [1, 2, 0, 2, 0, 3]]> : tensor<4x6xui32>
    %c_0 = stablehlo.constant dense<[[-1, 2, 0, 0, -3, -4], [2, 0, -1, 1, 0, 0], [0, -4, 1, -1, 6, 2], [-2, 0, 1, -1, 3, 2]]> : tensor<4x6xi32>
    return %c : tensor<4x6xui32>
  }
  func.func private @expected() -> (tensor<6xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[9, 13, 10, 9, 7, 14]> : tensor<6xui32>
    return %c : tensor<6xui32>
  }
}
