// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xui32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xui32>
    %1 = call @expected() : () -> tensor<8x9xui32>
    %2 = call @cumsum(%0) : (tensor<8x9xui32>) -> tensor<8x9xui32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xui32>, tensor<8x9xui32>) -> ()
    return %2 : tensor<8x9xui32>
  }
  func.func private @inputs() -> (tensor<8x9xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 1, 0, 2, 0, 1, 0, 2, 3], [0, 0, 0, 7, 0, 4, 2, 3, 5], [0, 4, 0, 0, 3, 0, 0, 0, 4], [0, 2, 5, 2, 0, 1, 0, 4, 0], [1, 4, 4, 1, 3, 5, 2, 3, 1], [5, 0, 5, 4, 0, 2, 2, 1, 2], [1, 2, 4, 0, 1, 3, 1, 6, 2], [1, 3, 0, 5, 3, 3, 5, 6, 0]]> : tensor<8x9xui32>
    return %c : tensor<8x9xui32>
  }
  func.func private @expected() -> (tensor<8x9xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 1, 0, 2, 0, 1, 0, 2, 3], [3, 1, 0, 9, 0, 5, 2, 5, 8], [3, 5, 0, 9, 3, 5, 2, 5, 12], [3, 7, 5, 11, 3, 6, 2, 9, 12], [4, 11, 9, 12, 6, 11, 4, 12, 13], [9, 11, 14, 16, 6, 13, 6, 13, 15], [10, 13, 18, 16, 7, 16, 7, 19, 17], [11, 16, 18, 21, 10, 19, 12, 25, 17]]> : tensor<8x9xui32>
    return %c : tensor<8x9xui32>
  }
  func.func private @cumsum(%arg0: tensor<8x9xui32>) -> tensor<8x9xui32> {
    %c = stablehlo.constant dense<0> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<ui32>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui32>, %arg2: tensor<ui32>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<ui32>
      stablehlo.return %2 : tensor<ui32>
    }) : (tensor<8x9xui32>, tensor<ui32>) -> tensor<8x9xui32>
    return %1 : tensor<8x9xui32>
  }
}
