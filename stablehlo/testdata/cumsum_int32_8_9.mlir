// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xi32>
    %1 = call @expected() : () -> tensor<8x9xi32>
    %2 = call @cumsum(%0) : (tensor<8x9xi32>) -> tensor<8x9xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xi32>, tensor<8x9xi32>) -> ()
    return %2 : tensor<8x9xi32>
  }
  func.func private @inputs() -> (tensor<8x9xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, -1, -4, 0, -1, 1, -3, -5, 0], [1, -2, 0, -2, -3, 2, 4, 2, 0], [0, -6, -3, 1, 2, 0, -2, -3, 0], [0, -1, -2, 0, 3, 3, 4, -3, -3], [2, 3, 0, 3, -1, 0, -4, 1, -2], [2, 1, -7, -1, 1, -2, 2, -2, -1], [3, -2, -2, -2, 0, -3, 3, 1, -3], [5, 1, -4, -1, -2, 0, 0, -2, -5]]> : tensor<8x9xi32>
    return %c : tensor<8x9xi32>
  }
  func.func private @expected() -> (tensor<8x9xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, -1, -4, 0, -1, 1, -3, -5, 0], [3, -3, -4, -2, -4, 3, 1, -3, 0], [3, -9, -7, -1, -2, 3, -1, -6, 0], [3, -10, -9, -1, 1, 6, 3, -9, -3], [5, -7, -9, 2, 0, 6, -1, -8, -5], [7, -6, -16, 1, 1, 4, 1, -10, -6], [10, -8, -18, -1, 1, 1, 4, -9, -9], [15, -7, -22, -2, -1, 1, 4, -11, -14]]> : tensor<8x9xi32>
    return %c : tensor<8x9xi32>
  }
  func.func private @cumsum(%arg0: tensor<8x9xi32>) -> tensor<8x9xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) : (tensor<8x9xi32>, tensor<i32>) -> tensor<8x9xi32>
    return %1 : tensor<8x9xi32>
  }
}
