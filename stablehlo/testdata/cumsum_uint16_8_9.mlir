// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xui16>
    %1 = call @expected() : () -> tensor<8x9xui16>
    %2 = call @cumsum(%0) : (tensor<8x9xui16>) -> tensor<8x9xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xui16>, tensor<8x9xui16>) -> ()
    return %2 : tensor<8x9xui16>
  }
  func.func private @inputs() -> (tensor<8x9xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 5, 3, 1, 7, 3, 5, 4, 7], [4, 1, 3, 6, 0, 2, 2, 0, 6], [2, 0, 0, 2, 0, 1, 0, 3, 0], [3, 0, 4, 6, 2, 4, 2, 0, 0], [1, 5, 0, 4, 4, 3, 0, 3, 1], [2, 1, 1, 0, 2, 0, 0, 0, 4], [0, 3, 3, 5, 1, 4, 3, 3, 2], [4, 2, 1, 1, 0, 5, 2, 1, 4]]> : tensor<8x9xui16>
    return %c : tensor<8x9xui16>
  }
  func.func private @expected() -> (tensor<8x9xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 5, 3, 1, 7, 3, 5, 4, 7], [5, 6, 6, 7, 7, 5, 7, 4, 13], [7, 6, 6, 9, 7, 6, 7, 7, 13], [10, 6, 10, 15, 9, 10, 9, 7, 13], [11, 11, 10, 19, 13, 13, 9, 10, 14], [13, 12, 11, 19, 15, 13, 9, 10, 18], [13, 15, 14, 24, 16, 17, 12, 13, 20], [17, 17, 15, 25, 16, 22, 14, 14, 24]]> : tensor<8x9xui16>
    return %c : tensor<8x9xui16>
  }
  func.func private @cumsum(%arg0: tensor<8x9xui16>) -> tensor<8x9xui16> {
    %c = stablehlo.constant dense<0> : tensor<ui16>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui16>) -> tensor<ui16>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui16>, %arg2: tensor<ui16>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<ui16>
      stablehlo.return %2 : tensor<ui16>
    }) : (tensor<8x9xui16>, tensor<ui16>) -> tensor<8x9xui16>
    return %1 : tensor<8x9xui16>
  }
}
