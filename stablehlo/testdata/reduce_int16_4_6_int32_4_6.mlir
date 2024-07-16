// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xi16> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xi16>
    %1 = call @expected() : () -> tensor<6xi16>
    %c = stablehlo.constant dense<3> : tensor<i16>
    %2 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<4x6xi16>, tensor<i16>) -> tensor<6xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<6xi16>, tensor<6xi16>) -> ()
    return %2 : tensor<6xi16>
  }
  func.func private @inputs() -> (tensor<4x6xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 2, -1, 0, 0, 5], [0, -2, 0, -2, 0, 1], [-5, -3, 2, 5, -1, -6], [0, 3, 0, 0, -3, -2]]> : tensor<4x6xi16>
    %c_0 = stablehlo.constant dense<[[4, 3, -2, 2, 1, 4], [2, 0, 0, 0, 0, 3], [1, 0, 5, 1, 0, 2], [-2, 0, 10, 1, 0, -2]]> : tensor<4x6xi32>
    return %c : tensor<4x6xi16>
  }
  func.func private @expected() -> (tensor<6xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[-1, 3, 4, 6, -1, 1]> : tensor<6xi16>
    return %c : tensor<6xi16>
  }
}
