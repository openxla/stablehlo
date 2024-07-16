// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xui64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui32>, tensor<3x6xui64>)
    %1 = call @expected() : () -> tensor<4x6xui64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui32>) -> tensor<4x3xui64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xui64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xui64>, tensor<3x6xui64>) -> tensor<4x6xui64>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xui64>, tensor<4x6xui64>) -> ()
    return %4 : tensor<4x6xui64>
  }
  func.func private @inputs() -> (tensor<4x3xui32> {mhlo.layout_mode = "default"}, tensor<3x6xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 10, 3], [4, 2, 4], [1, 1, 0], [1, 3, 1]]> : tensor<4x3xui32>
    %c_0 = stablehlo.constant dense<[[0, 1, 0, 4, 0, 3], [1, 5, 5, 1, 6, 1], [4, 3, 3, 1, 0, 0]]> : tensor<3x6xui64>
    return %c, %c_0 : tensor<4x3xui32>, tensor<3x6xui64>
  }
  func.func private @expected() -> (tensor<4x6xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[22, 60, 59, 17, 60, 13], [18, 26, 22, 22, 12, 14], [1, 6, 5, 5, 6, 4], [7, 19, 18, 8, 18, 6]]> : tensor<4x6xui64>
    return %c : tensor<4x6xui64>
  }
}
