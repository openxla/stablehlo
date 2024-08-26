// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xi64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xi64>
    %1 = call @expected() : () -> tensor<3x5xi64>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %2 = "stablehlo.reduce_window"(%0, %c) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i64>
      stablehlo.return %3 : tensor<i64>
    }) : (tensor<4x6xi64>, tensor<i64>) -> tensor<3x5xi64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3x5xi64>, tensor<3x5xi64>) -> ()
    return %2 : tensor<3x5xi64>
  }
  func.func private @inputs() -> (tensor<4x6xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-1, -5, 0, 1, -2, 0], [3, 1, -2, 4, 0, 1], [0, 0, -1, 4, 1, -7], [3, -3, -3, -2, 0, -5]]> : tensor<4x6xi64>
    return %c : tensor<4x6xi64>
  }
  func.func private @expected() -> (tensor<3x5xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 1, 4, 4, 1], [3, 1, 4, 4, 1], [3, 1, 4, 4, 1]]> : tensor<3x5xi64>
    return %c : tensor<3x5xi64>
  }
}
