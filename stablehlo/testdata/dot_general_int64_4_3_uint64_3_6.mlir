// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xi64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi64>, tensor<3x6xui64>)
    %1 = call @expected() : () -> tensor<4x6xi64>
    %2 = stablehlo.convert %0#0 : tensor<4x3xi64>
    %3 = stablehlo.convert %0#1 : (tensor<3x6xui64>) -> tensor<3x6xi64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xi64>, tensor<3x6xi64>) -> tensor<4x6xi64>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xi64>, tensor<4x6xi64>) -> ()
    return %4 : tensor<4x6xi64>
  }
  func.func private @inputs() -> (tensor<4x3xi64> {mhlo.layout_mode = "default"}, tensor<3x6xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 3, -5], [3, -3, 7], [-3, 1, 0], [0, 5, -3]]> : tensor<4x3xi64>
    %c_0 = stablehlo.constant dense<[[3, 3, 1, 0, 1, 2], [1, 2, 5, 2, 1, 2], [2, 6, 1, 2, 0, 3]]> : tensor<3x6xui64>
    return %c, %c_0 : tensor<4x3xi64>, tensor<3x6xui64>
  }
  func.func private @expected() -> (tensor<4x6xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-7, -24, 10, -4, 3, -9], [20, 45, -5, 8, 0, 21], [-8, -7, 2, 2, -2, -4], [-1, -8, 22, 4, 5, 1]]> : tensor<4x6xi64>
    return %c : tensor<4x6xi64>
  }
}
