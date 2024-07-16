// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xui8>
    %1 = call @expected() : () -> tensor<8x9xui8>
    %2 = call @cummin(%0) : (tensor<8x9xui8>) -> tensor<8x9xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<8x9xui8>, tensor<8x9xui8>) -> ()
    return %2 : tensor<8x9xui8>
  }
  func.func private @inputs() -> (tensor<8x9xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, 0, 1, 2, 0, 2, 1, 0, 0], [5, 6, 1, 0, 2, 6, 4, 0, 0], [0, 4, 0, 0, 3, 3, 1, 0, 0], [2, 1, 0, 3, 2, 3, 1, 3, 3], [2, 0, 2, 0, 1, 0, 1, 5, 1], [6, 3, 0, 1, 1, 0, 2, 1, 1], [1, 5, 2, 1, 1, 2, 1, 4, 6], [4, 5, 0, 2, 7, 6, 1, 0, 0]]> : tensor<8x9xui8>
    return %c : tensor<8x9xui8>
  }
  func.func private @expected() -> (tensor<8x9xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, 0, 1, 2, 0, 2, 1, 0, 0], [4, 0, 1, 0, 0, 2, 1, 0, 0], [0, 0, 0, 0, 0, 2, 1, 0, 0], [0, 0, 0, 0, 0, 2, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]]> : tensor<8x9xui8>
    return %c : tensor<8x9xui8>
  }
  func.func private @cummin(%arg0: tensor<8x9xui8>) -> tensor<8x9xui8> {
    %c = stablehlo.constant dense<255> : tensor<ui8>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui8>) -> tensor<ui8>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui8>, %arg2: tensor<ui8>):
      %2 = stablehlo.minimum %arg1, %arg2 : tensor<ui8>
      stablehlo.return %2 : tensor<ui8>
    }) : (tensor<8x9xui8>, tensor<ui8>) -> tensor<8x9xui8>
    return %1 : tensor<8x9xui8>
  }
}
