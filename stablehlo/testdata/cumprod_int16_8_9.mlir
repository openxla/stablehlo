// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xi16>
    %1 = call @expected() : () -> tensor<8x9xi16>
    %2 = call @cumprod(%0) : (tensor<8x9xi16>) -> tensor<8x9xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xi16>, tensor<8x9xi16>) -> ()
    return %2 : tensor<8x9xi16>
  }
  func.func private @inputs() -> (tensor<8x9xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-5, -4, -1, -1, -5, 0, -4, 0, 8], [-3, 4, 1, 0, -3, -1, 3, 1, 8], [-1, 0, 0, 0, 1, 0, 3, 2, 1], [-2, -3, 1, -1, 4, 0, 0, -1, 1], [5, 0, 1, 3, -6, 0, -4, 2, -2], [1, -6, 0, 0, 2, 0, 7, -4, -2], [-3, 0, 0, 4, 1, -1, 1, 2, -1], [-1, -1, 0, 0, -1, 3, 6, 0, 2]]> : tensor<8x9xi16>
    return %c : tensor<8x9xi16>
  }
  func.func private @expected() -> (tensor<8x9xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-5, -4, -1, -1, -5, 0, -4, 0, 8], [15, -16, -1, 0, 15, 0, -12, 0, 64], [-15, 0, 0, 0, 15, 0, -36, 0, 64], [30, 0, 0, 0, 60, 0, 0, 0, 64], [150, 0, 0, 0, -360, 0, 0, 0, -128], [150, 0, 0, 0, -720, 0, 0, 0, 256], [-450, 0, 0, 0, -720, 0, 0, 0, -256], [450, 0, 0, 0, 720, 0, 0, 0, -512]]> : tensor<8x9xi16>
    return %c : tensor<8x9xi16>
  }
  func.func private @cumprod(%arg0: tensor<8x9xi16>) -> tensor<8x9xi16> {
    %c = stablehlo.constant dense<1> : tensor<i16>
    %0 = "stablehlo.reduce_window"(%arg0, %c) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i16>, %arg2: tensor<i16>):
      %1 = stablehlo.multiply %arg1, %arg2 : tensor<i16>
      stablehlo.return %1 : tensor<i16>
    }) : (tensor<8x9xi16>, tensor<i16>) -> tensor<8x9xi16>
    return %0 : tensor<8x9xi16>
  }
}
