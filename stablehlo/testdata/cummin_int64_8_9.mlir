// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xi64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xi64>
    %1 = call @expected() : () -> tensor<8x9xi64>
    %2 = call @cummin(%0) : (tensor<8x9xi64>) -> tensor<8x9xi64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xi64>, tensor<8x9xi64>) -> ()
    return %2 : tensor<8x9xi64>
  }
  func.func private @inputs() -> (tensor<8x9xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-2, -2, -1, -1, 0, 0, -5, 4, 5], [0, 0, 0, 0, 0, 1, -2, 4, 4], [2, 3, 1, 1, 7, -2, 0, -3, 1], [-4, -1, 4, -3, -5, 0, 2, -1, 5], [-2, 0, 0, 0, 1, 0, 1, 2, 1], [-1, -2, 0, 1, 2, 0, 2, 0, 1], [-1, 0, 1, -1, 0, -2, -3, -3, 3], [0, -1, 3, 0, 2, 0, 0, -2, 0]]> : tensor<8x9xi64>
    return %c : tensor<8x9xi64>
  }
  func.func private @expected() -> (tensor<8x9xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-2, -2, -1, -1, 0, 0, -5, 4, 5], [-2, -2, -1, -1, 0, 0, -5, 4, 4], [-2, -2, -1, -1, 0, -2, -5, -3, 1], [-4, -2, -1, -3, -5, -2, -5, -3, 1], [-4, -2, -1, -3, -5, -2, -5, -3, 1], [-4, -2, -1, -3, -5, -2, -5, -3, 1], [-4, -2, -1, -3, -5, -2, -5, -3, 1], [-4, -2, -1, -3, -5, -2, -5, -3, 0]]> : tensor<8x9xi64>
    return %c : tensor<8x9xi64>
  }
  func.func private @cummin(%arg0: tensor<8x9xi64>) -> tensor<8x9xi64> {
    %c = stablehlo.constant dense<9223372036854775807> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<i64>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %2 = stablehlo.minimum %arg1, %arg2 : tensor<i64>
      stablehlo.return %2 : tensor<i64>
    }) : (tensor<8x9xi64>, tensor<i64>) -> tensor<8x9xi64>
    return %1 : tensor<8x9xi64>
  }
}
