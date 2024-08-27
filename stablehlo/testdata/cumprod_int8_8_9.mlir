// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xi8>
    %1 = call @expected() : () -> tensor<8x9xi8>
    %2 = call @cumprod(%0) : (tensor<8x9xi8>) -> tensor<8x9xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xi8>, tensor<8x9xi8>) -> ()
    return %2 : tensor<8x9xi8>
  }
  func.func private @inputs() -> (tensor<8x9xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, -6, -1, 1, 1, 3, 2, -1, 3], [-3, -1, 1, 6, 1, -1, -2, 0, 0], [-2, 2, 2, 1, -5, 2, -2, -6, 2], [-6, 1, -2, -1, 0, 5, 5, 1, -3], [5, 0, 2, -3, 3, 1, 0, -1, -2], [1, 0, -3, 3, -1, -1, 0, -4, 7], [2, 0, -1, -4, 0, 0, 0, 0, -3], [-2, 0, -1, 3, 2, -2, 0, 0, 0]]> : tensor<8x9xi8>
    return %c : tensor<8x9xi8>
  }
  func.func private @expected() -> (tensor<8x9xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, -6, -1, 1, 1, 3, 2, -1, 3], [-12, 6, -1, 6, 1, -3, -4, 0, 0], [24, 12, -2, 6, -5, -6, 8, 0, 0], [112, 12, 4, -6, 0, -30, 40, 0, 0], [48, 0, 8, 18, 0, -30, 0, 0, 0], [48, 0, -24, 54, 0, 30, 0, 0, 0], [96, 0, 24, 40, 0, 0, 0, 0, 0], [64, 0, -24, 120, 0, 0, 0, 0, 0]]> : tensor<8x9xi8>
    return %c : tensor<8x9xi8>
  }
  func.func private @cumprod(%arg0: tensor<8x9xi8>) -> tensor<8x9xi8> {
    %c = stablehlo.constant dense<1> : tensor<i8>
    %0 = "stablehlo.reduce_window"(%arg0, %c) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i8>, %arg2: tensor<i8>):
      %1 = stablehlo.multiply %arg1, %arg2 : tensor<i8>
      stablehlo.return %1 : tensor<i8>
    }) : (tensor<8x9xi8>, tensor<i8>) -> tensor<8x9xi8>
    return %0 : tensor<8x9xi8>
  }
}
