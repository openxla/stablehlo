// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xi32>
    %1 = call @expected() : () -> tensor<8x9xi32>
    %2 = call @cumprod(%0) : (tensor<8x9xi32>) -> tensor<8x9xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xi32>, tensor<8x9xi32>) -> ()
    return %2 : tensor<8x9xi32>
  }
  func.func private @inputs() -> (tensor<8x9xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, -2, -1, 0, -4, 0, 1, -3, -1], [1, -1, 0, -3, 2, -2, 1, 1, 0], [-1, 1, -2, 2, 0, 0, 4, 2, 0], [-4, 3, -1, 1, -2, -2, -2, 4, 5], [5, 2, 0, 0, 0, 1, -1, 0, 0], [2, 1, -3, -4, -1, -6, -1, 3, 1], [0, 3, 5, -2, 0, 3, 0, -4, 5], [2, -4, 2, 0, 3, -6, 1, -3, 4]]> : tensor<8x9xi32>
    return %c : tensor<8x9xi32>
  }
  func.func private @expected() -> (tensor<8x9xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, -2, -1, 0, -4, 0, 1, -3, -1], [3, 2, 0, 0, -8, 0, 1, -3, 0], [-3, 2, 0, 0, 0, 0, 4, -6, 0], [12, 6, 0, 0, 0, 0, -8, -24, 0], [60, 12, 0, 0, 0, 0, 8, 0, 0], [120, 12, 0, 0, 0, 0, -8, 0, 0], [0, 36, 0, 0, 0, 0, 0, 0, 0], [0, -144, 0, 0, 0, 0, 0, 0, 0]]> : tensor<8x9xi32>
    return %c : tensor<8x9xi32>
  }
  func.func private @cumprod(%arg0: tensor<8x9xi32>) -> tensor<8x9xi32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = "stablehlo.reduce_window"(%arg0, %c) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.multiply %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
    }) : (tensor<8x9xi32>, tensor<i32>) -> tensor<8x9xi32>
    return %0 : tensor<8x9xi32>
  }
}
