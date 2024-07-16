// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xi32>
    %1 = call @expected() : () -> tensor<8x9xi32>
    %2 = call @cummax(%0) : (tensor<8x9xi32>) -> tensor<8x9xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xi32>, tensor<8x9xi32>) -> ()
    return %2 : tensor<8x9xi32>
  }
  func.func private @inputs() -> (tensor<8x9xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-2, 1, 6, 0, 1, 0, 2, 0, -1], [-3, 4, 1, 2, 6, 3, 0, 0, -4], [0, 0, 0, 1, -3, -2, 1, 1, 0], [-4, 1, -4, -1, 4, 2, 0, -2, 2], [2, -3, -6, -1, 1, 3, 2, 1, 1], [4, 1, -4, 4, 3, -1, 1, -2, -6], [0, 2, 2, 1, -2, 0, -2, 0, 3], [0, 4, -7, 6, 2, 4, 1, -1, -3]]> : tensor<8x9xi32>
    return %c : tensor<8x9xi32>
  }
  func.func private @expected() -> (tensor<8x9xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-2, 1, 6, 0, 1, 0, 2, 0, -1], [-2, 4, 6, 2, 6, 3, 2, 0, -1], [0, 4, 6, 2, 6, 3, 2, 1, 0], [0, 4, 6, 2, 6, 3, 2, 1, 2], [2, 4, 6, 2, 6, 3, 2, 1, 2], [4, 4, 6, 4, 6, 3, 2, 1, 2], [4, 4, 6, 4, 6, 3, 2, 1, 3], [4, 4, 6, 6, 6, 4, 2, 1, 3]]> : tensor<8x9xi32>
    return %c : tensor<8x9xi32>
  }
  func.func private @cummax(%arg0: tensor<8x9xi32>) -> tensor<8x9xi32> {
    %c = stablehlo.constant dense<-2147483648> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %2 = stablehlo.maximum %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) : (tensor<8x9xi32>, tensor<i32>) -> tensor<8x9xi32>
    return %1 : tensor<8x9xi32>
  }
}
