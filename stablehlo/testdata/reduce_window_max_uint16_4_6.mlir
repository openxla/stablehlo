// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xui16>
    %1 = call @expected() : () -> tensor<3x5xui16>
    %c = stablehlo.constant dense<1> : tensor<ui16>
    %2 = "stablehlo.reduce_window"(%0, %c) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %3 : tensor<ui16>
    }) : (tensor<4x6xui16>, tensor<ui16>) -> tensor<3x5xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3x5xui16>, tensor<3x5xui16>) -> ()
    return %2 : tensor<3x5xui16>
  }
  func.func private @inputs() -> (tensor<4x6xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 0, 0, 0, 4, 0], [0, 4, 5, 2, 3, 1], [2, 4, 3, 3, 0, 2], [4, 3, 0, 0, 3, 0]]> : tensor<4x6xui16>
    return %c : tensor<4x6xui16>
  }
  func.func private @expected() -> (tensor<3x5xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, 5, 5, 4, 4], [4, 5, 5, 3, 3], [4, 4, 3, 3, 3]]> : tensor<3x5xui16>
    return %c : tensor<3x5xui16>
  }
}
