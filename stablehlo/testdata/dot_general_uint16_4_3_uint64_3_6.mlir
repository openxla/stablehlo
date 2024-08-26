// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xui64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui16>, tensor<3x6xui64>)
    %1 = call @expected() : () -> tensor<4x6xui64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui16>) -> tensor<4x3xui64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xui64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xui64>, tensor<3x6xui64>) -> tensor<4x6xui64>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xui64>, tensor<4x6xui64>) -> ()
    return %4 : tensor<4x6xui64>
  }
  func.func private @inputs() -> (tensor<4x3xui16> {mhlo.layout_mode = "default"}, tensor<3x6xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 0, 3], [1, 1, 2], [1, 1, 1], [1, 1, 9]]> : tensor<4x3xui16>
    %c_0 = stablehlo.constant dense<[[1, 0, 0, 1, 2, 3], [2, 1, 1, 3, 1, 2], [1, 1, 0, 1, 4, 1]]> : tensor<3x6xui64>
    return %c, %c_0 : tensor<4x3xui16>, tensor<3x6xui64>
  }
  func.func private @expected() -> (tensor<4x6xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[6, 3, 0, 6, 18, 12], [5, 3, 1, 6, 11, 7], [4, 2, 1, 5, 7, 6], [12, 10, 1, 13, 39, 14]]> : tensor<4x6xui64>
    return %c : tensor<4x6xui64>
  }
}
