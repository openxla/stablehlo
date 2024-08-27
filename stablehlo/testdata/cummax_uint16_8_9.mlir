// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xui16>
    %1 = call @expected() : () -> tensor<8x9xui16>
    %2 = call @cummax(%0) : (tensor<8x9xui16>) -> tensor<8x9xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xui16>, tensor<8x9xui16>) -> ()
    return %2 : tensor<8x9xui16>
  }
  func.func private @inputs() -> (tensor<8x9xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 3, 0, 2, 5, 0, 0, 1, 0], [3, 0, 3, 3, 3, 1, 1, 2, 6], [2, 0, 1, 2, 3, 4, 2, 0, 2], [2, 0, 1, 3, 3, 2, 3, 2, 1], [2, 2, 2, 3, 2, 1, 4, 3, 1], [0, 2, 1, 2, 5, 1, 0, 4, 0], [0, 7, 0, 2, 0, 2, 3, 1, 1], [1, 0, 2, 1, 1, 0, 0, 1, 0]]> : tensor<8x9xui16>
    return %c : tensor<8x9xui16>
  }
  func.func private @expected() -> (tensor<8x9xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 3, 0, 2, 5, 0, 0, 1, 0], [3, 3, 3, 3, 5, 1, 1, 2, 6], [3, 3, 3, 3, 5, 4, 2, 2, 6], [3, 3, 3, 3, 5, 4, 3, 2, 6], [3, 3, 3, 3, 5, 4, 4, 3, 6], [3, 3, 3, 3, 5, 4, 4, 4, 6], [3, 7, 3, 3, 5, 4, 4, 4, 6], [3, 7, 3, 3, 5, 4, 4, 4, 6]]> : tensor<8x9xui16>
    return %c : tensor<8x9xui16>
  }
  func.func private @cummax(%arg0: tensor<8x9xui16>) -> tensor<8x9xui16> {
    %c = stablehlo.constant dense<0> : tensor<ui16>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui16>) -> tensor<ui16>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui16>, %arg2: tensor<ui16>):
      %2 = stablehlo.maximum %arg1, %arg2 : tensor<ui16>
      stablehlo.return %2 : tensor<ui16>
    }) : (tensor<8x9xui16>, tensor<ui16>) -> tensor<8x9xui16>
    return %1 : tensor<8x9xui16>
  }
}
