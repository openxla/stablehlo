// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xui16> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xui16>
    %1 = call @expected() : () -> tensor<6xui16>
    %c = stablehlo.constant dense<3> : tensor<ui16>
    %2 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<4x6xui16>, tensor<ui16>) -> tensor<6xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<6xui16>, tensor<6xui16>) -> ()
    return %2 : tensor<6xui16>
  }
  func.func private @inputs() -> (tensor<4x6xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 5, 4, 0, 1, 5], [7, 0, 1, 0, 1, 2], [3, 1, 0, 4, 3, 1], [2, 3, 0, 3, 5, 0]]> : tensor<4x6xui16>
    %c_0 = stablehlo.constant dense<[[-1, 6, 3, 0, -2, 2], [-1, -6, -3, 3, 0, -4], [-2, 5, -1, 3, -3, 4], [0, -4, -1, 4, 0, 0]]> : tensor<4x6xi32>
    return %c : tensor<4x6xui16>
  }
  func.func private @expected() -> (tensor<6xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[17, 12, 8, 10, 13, 11]> : tensor<6xui16>
    return %c : tensor<6xui16>
  }
}
