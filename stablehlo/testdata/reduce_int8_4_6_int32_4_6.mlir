// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xi8> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xi8>
    %1 = call @expected() : () -> tensor<6xi8>
    %c = stablehlo.constant dense<3> : tensor<i8>
    %2 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<4x6xi8>, tensor<i8>) -> tensor<6xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<6xi8>, tensor<6xi8>) -> ()
    return %2 : tensor<6xi8>
  }
  func.func private @inputs() -> (tensor<4x6xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, -1, 0, 0, 2, 1], [1, 0, -1, -4, 1, -4], [2, 1, 1, -3, 0, 6], [0, 0, -3, 3, 6, 0]]> : tensor<4x6xi8>
    %c_0 = stablehlo.constant dense<[[-3, 1, 2, 2, -1, -1], [1, -1, -1, 3, 0, -2], [-3, 0, 0, -2, -2, 1], [5, 1, 1, -3, -3, 1]]> : tensor<4x6xi32>
    return %c : tensor<4x6xi8>
  }
  func.func private @expected() -> (tensor<6xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[8, 3, 0, -1, 12, 6]> : tensor<6xi8>
    return %c : tensor<6xi8>
  }
}
