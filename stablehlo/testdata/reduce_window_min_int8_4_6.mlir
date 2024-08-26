// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xi8>
    %1 = call @expected() : () -> tensor<3x5xi8>
    %c = stablehlo.constant dense<127> : tensor<i8>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i8>) -> tensor<i8>
    %3 = "stablehlo.reduce_window"(%0, %2) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %4 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %4 : tensor<i8>
    }) : (tensor<4x6xi8>, tensor<i8>) -> tensor<3x5xi8>
    stablehlo.custom_call @check.expect_eq(%3, %1) {has_side_effect = true} : (tensor<3x5xi8>, tensor<3x5xi8>) -> ()
    return %3 : tensor<3x5xi8>
  }
  func.func private @inputs() -> (tensor<4x6xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-3, -2, 2, 0, -3, 0], [0, 0, 0, 1, 3, -2], [-4, 0, -5, -2, 3, -3], [-7, -1, 0, -2, 1, 1]]> : tensor<4x6xi8>
    return %c : tensor<4x6xi8>
  }
  func.func private @expected() -> (tensor<3x5xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-3, -2, 0, -3, -3], [-4, -5, -5, -2, -3], [-7, -5, -5, -2, -3]]> : tensor<3x5xi8>
    return %c : tensor<3x5xi8>
  }
}
