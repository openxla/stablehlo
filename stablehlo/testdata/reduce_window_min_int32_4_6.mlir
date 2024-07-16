// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xi32>
    %1 = call @expected() : () -> tensor<3x5xi32>
    %c = stablehlo.constant dense<2147483647> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
    %3 = "stablehlo.reduce_window"(%0, %2) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %4 = stablehlo.minimum %arg0, %arg1 : tensor<i32>
      stablehlo.return %4 : tensor<i32>
    }) : (tensor<4x6xi32>, tensor<i32>) -> tensor<3x5xi32>
    stablehlo.custom_call @check.expect_eq(%3, %1) {has_side_effect = true} : (tensor<3x5xi32>, tensor<3x5xi32>) -> ()
    return %3 : tensor<3x5xi32>
  }
  func.func private @inputs() -> (tensor<4x6xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, -1, 1, 1, 2, 0], [1, 0, 0, 0, -1, -1], [0, -4, -1, 0, -4, -4], [-2, -1, 0, 5, 3, 2]]> : tensor<4x6xi32>
    return %c : tensor<4x6xi32>
  }
  func.func private @expected() -> (tensor<3x5xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-1, -1, 0, -1, -1], [-4, -4, -1, -4, -4], [-4, -4, -1, -4, -4]]> : tensor<3x5xi32>
    return %c : tensor<3x5xi32>
  }
}
