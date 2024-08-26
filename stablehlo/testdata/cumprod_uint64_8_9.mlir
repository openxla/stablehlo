// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xui64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xui64>
    %1 = call @expected() : () -> tensor<8x9xui64>
    %2 = call @cumprod(%0) : (tensor<8x9xui64>) -> tensor<8x9xui64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xui64>, tensor<8x9xui64>) -> ()
    return %2 : tensor<8x9xui64>
  }
  func.func private @inputs() -> (tensor<8x9xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[7, 0, 2, 1, 1, 0, 1, 1, 0], [3, 0, 2, 0, 1, 3, 1, 2, 0], [2, 5, 1, 1, 2, 0, 9, 5, 2], [3, 1, 2, 3, 0, 0, 0, 0, 3], [4, 3, 3, 7, 2, 6, 2, 3, 1], [4, 4, 3, 4, 3, 1, 0, 0, 2], [0, 5, 1, 0, 3, 3, 3, 2, 4], [3, 1, 4, 0, 3, 3, 2, 1, 4]]> : tensor<8x9xui64>
    return %c : tensor<8x9xui64>
  }
  func.func private @expected() -> (tensor<8x9xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[7, 0, 2, 1, 1, 0, 1, 1, 0], [21, 0, 4, 0, 1, 0, 1, 2, 0], [42, 0, 4, 0, 2, 0, 9, 10, 0], [126, 0, 8, 0, 0, 0, 0, 0, 0], [504, 0, 24, 0, 0, 0, 0, 0, 0], [2016, 0, 72, 0, 0, 0, 0, 0, 0], [0, 0, 72, 0, 0, 0, 0, 0, 0], [0, 0, 288, 0, 0, 0, 0, 0, 0]]> : tensor<8x9xui64>
    return %c : tensor<8x9xui64>
  }
  func.func private @cumprod(%arg0: tensor<8x9xui64>) -> tensor<8x9xui64> {
    %c = stablehlo.constant dense<1> : tensor<ui64>
    %0 = "stablehlo.reduce_window"(%arg0, %c) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui64>, %arg2: tensor<ui64>):
      %1 = stablehlo.multiply %arg1, %arg2 : tensor<ui64>
      stablehlo.return %1 : tensor<ui64>
    }) : (tensor<8x9xui64>, tensor<ui64>) -> tensor<8x9xui64>
    return %0 : tensor<8x9xui64>
  }
}
