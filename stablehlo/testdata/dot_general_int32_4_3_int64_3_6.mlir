// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xi64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi32>, tensor<3x6xi64>)
    %1 = call @expected() : () -> tensor<4x6xi64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi32>) -> tensor<4x3xi64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xi64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xi64>, tensor<3x6xi64>) -> tensor<4x6xi64>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xi64>, tensor<4x6xi64>) -> ()
    return %4 : tensor<4x6xi64>
  }
  func.func private @inputs() -> (tensor<4x3xi32> {mhlo.layout_mode = "default"}, tensor<3x6xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, -5, 1], [-1, -7, 4], [6, -4, 0], [-2, -1, -1]]> : tensor<4x3xi32>
    %c_0 = stablehlo.constant dense<[[-1, -1, 1, 6, 0, 1], [0, -4, -6, 2, -4, -1], [3, -2, -1, 0, 0, 2]]> : tensor<3x6xi64>
    return %c, %c_0 : tensor<4x3xi32>, tensor<3x6xi64>
  }
  func.func private @expected() -> (tensor<4x6xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 18, 29, -10, 20, 7], [13, 21, 37, -20, 28, 14], [-6, 10, 30, 28, 16, 10], [-1, 8, 5, -14, 4, -3]]> : tensor<4x6xi64>
    return %c : tensor<4x6xi64>
  }
}
