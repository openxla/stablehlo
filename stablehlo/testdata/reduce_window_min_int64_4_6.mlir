// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xi64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xi64>
    %1 = call @expected() : () -> tensor<3x5xi64>
    %c = stablehlo.constant dense<9223372036854775807> : tensor<i64>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<i64>
    %3 = "stablehlo.reduce_window"(%0, %2) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %4 = stablehlo.minimum %arg0, %arg1 : tensor<i64>
      stablehlo.return %4 : tensor<i64>
    }) : (tensor<4x6xi64>, tensor<i64>) -> tensor<3x5xi64>
    stablehlo.custom_call @check.expect_eq(%3, %1) {has_side_effect = true} : (tensor<3x5xi64>, tensor<3x5xi64>) -> ()
    return %3 : tensor<3x5xi64>
  }
  func.func private @inputs() -> (tensor<4x6xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-2, 0, 0, 1, -2, 2], [1, -5, 0, 1, -3, -3], [-4, 2, -1, -4, 2, 0], [-2, 0, -5, 2, -5, 0]]> : tensor<4x6xi64>
    return %c : tensor<4x6xi64>
  }
  func.func private @expected() -> (tensor<3x5xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-5, -5, 0, -3, -3], [-5, -5, -4, -4, -3], [-4, -5, -5, -5, -5]]> : tensor<3x5xi64>
    return %c : tensor<3x5xi64>
  }
}
