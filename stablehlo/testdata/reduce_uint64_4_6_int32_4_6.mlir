// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xui64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xui64>
    %1 = call @expected() : () -> tensor<6xui64>
    %c = stablehlo.constant dense<3> : tensor<ui64>
    %2 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<4x6xui64>, tensor<ui64>) -> tensor<6xui64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<6xui64>, tensor<6xui64>) -> ()
    return %2 : tensor<6xui64>
  }
  func.func private @inputs() -> (tensor<4x6xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 4, 1, 1, 4, 2], [3, 1, 2, 1, 1, 5], [1, 1, 1, 3, 0, 1], [5, 5, 1, 3, 1, 1]]> : tensor<4x6xui64>
    %c_0 = stablehlo.constant dense<[[-4, -2, 2, 1, 0, 5], [0, 0, 0, 0, 0, -2], [-4, 3, 0, -6, 4, 1], [-1, 0, -2, -2, -4, 0]]> : tensor<4x6xi32>
    return %c : tensor<4x6xui64>
  }
  func.func private @expected() -> (tensor<6xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[12, 14, 8, 11, 9, 12]> : tensor<6xui64>
    return %c : tensor<6xui64>
  }
}
