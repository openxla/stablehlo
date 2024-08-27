// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xui8>
    %1 = call @expected() : () -> tensor<3x5xui8>
    %c = stablehlo.constant dense<1> : tensor<ui8>
    %2 = "stablehlo.reduce_window"(%0, %c) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<4x6xui8>, tensor<ui8>) -> tensor<3x5xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3x5xui8>, tensor<3x5xui8>) -> ()
    return %2 : tensor<3x5xui8>
  }
  func.func private @inputs() -> (tensor<4x6xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[5, 1, 2, 0, 1, 1], [0, 0, 2, 1, 0, 3], [1, 2, 0, 0, 2, 3], [0, 1, 0, 1, 3, 4]]> : tensor<4x6xui8>
    return %c : tensor<4x6xui8>
  }
  func.func private @expected() -> (tensor<3x5xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[5, 2, 2, 1, 3], [2, 2, 2, 2, 3], [2, 2, 1, 3, 4]]> : tensor<3x5xui8>
    return %c : tensor<3x5xui8>
  }
}
