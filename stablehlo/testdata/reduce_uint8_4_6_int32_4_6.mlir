// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xui8> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xui8>
    %1 = call @expected() : () -> tensor<6xui8>
    %c = stablehlo.constant dense<3> : tensor<ui8>
    %2 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<4x6xui8>, tensor<ui8>) -> tensor<6xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<6xui8>, tensor<6xui8>) -> ()
    return %2 : tensor<6xui8>
  }
  func.func private @inputs() -> (tensor<4x6xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1, 1, 3, 2, 6], [0, 2, 0, 2, 1, 0], [2, 1, 2, 1, 3, 1], [7, 1, 2, 4, 3, 2]]> : tensor<4x6xui8>
    %c_0 = stablehlo.constant dense<[[0, 1, 1, 0, -2, 0], [0, -2, -1, -2, 6, 2], [0, 1, 2, 0, 3, 0], [0, -1, 0, -5, -4, 3]]> : tensor<4x6xi32>
    return %c : tensor<4x6xui8>
  }
  func.func private @expected() -> (tensor<6xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[12, 8, 8, 13, 12, 12]> : tensor<6xui8>
    return %c : tensor<6xui8>
  }
}
