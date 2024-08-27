// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xui16>
    %1 = call @expected() : () -> tensor<8x9xui16>
    %2 = call @cumprod(%0) : (tensor<8x9xui16>) -> tensor<8x9xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xui16>, tensor<8x9xui16>) -> ()
    return %2 : tensor<8x9xui16>
  }
  func.func private @inputs() -> (tensor<8x9xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 0, 1, 5, 0, 3, 2, 1, 1], [0, 1, 0, 3, 5, 0, 3, 3, 1], [1, 5, 1, 0, 3, 2, 1, 4, 5], [1, 1, 2, 4, 4, 0, 4, 2, 1], [0, 3, 2, 1, 0, 5, 2, 2, 4], [3, 1, 2, 2, 1, 0, 0, 0, 2], [4, 1, 0, 3, 1, 3, 3, 4, 2], [0, 0, 2, 5, 0, 2, 0, 1, 0]]> : tensor<8x9xui16>
    return %c : tensor<8x9xui16>
  }
  func.func private @expected() -> (tensor<8x9xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 0, 1, 5, 0, 3, 2, 1, 1], [0, 0, 0, 15, 0, 0, 6, 3, 1], [0, 0, 0, 0, 0, 0, 6, 12, 5], [0, 0, 0, 0, 0, 0, 24, 24, 5], [0, 0, 0, 0, 0, 0, 48, 48, 20], [0, 0, 0, 0, 0, 0, 0, 0, 40], [0, 0, 0, 0, 0, 0, 0, 0, 80], [0, 0, 0, 0, 0, 0, 0, 0, 0]]> : tensor<8x9xui16>
    return %c : tensor<8x9xui16>
  }
  func.func private @cumprod(%arg0: tensor<8x9xui16>) -> tensor<8x9xui16> {
    %c = stablehlo.constant dense<1> : tensor<ui16>
    %0 = "stablehlo.reduce_window"(%arg0, %c) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui16>, %arg2: tensor<ui16>):
      %1 = stablehlo.multiply %arg1, %arg2 : tensor<ui16>
      stablehlo.return %1 : tensor<ui16>
    }) : (tensor<8x9xui16>, tensor<ui16>) -> tensor<8x9xui16>
    return %0 : tensor<8x9xui16>
  }
}
