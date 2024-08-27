// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xui32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui16>, tensor<3x6xui32>)
    %1 = call @expected() : () -> tensor<4x6xui32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui16>) -> tensor<4x3xui32>
    %3 = stablehlo.convert %0#1 : tensor<3x6xui32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xui32>, tensor<3x6xui32>) -> tensor<4x6xui32>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xui32>, tensor<4x6xui32>) -> ()
    return %4 : tensor<4x6xui32>
  }
  func.func private @inputs() -> (tensor<4x3xui16> {mhlo.layout_mode = "default"}, tensor<3x6xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 0, 1], [1, 4, 2], [3, 1, 2], [1, 2, 2]]> : tensor<4x3xui16>
    %c_0 = stablehlo.constant dense<[[0, 1, 1, 0, 2, 4], [3, 2, 2, 4, 3, 2], [3, 1, 3, 2, 0, 2]]> : tensor<3x6xui32>
    return %c, %c_0 : tensor<4x3xui16>, tensor<3x6xui32>
  }
  func.func private @expected() -> (tensor<4x6xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 2, 4, 2, 2, 6], [18, 11, 15, 20, 14, 16], [9, 7, 11, 8, 9, 18], [12, 7, 11, 12, 8, 12]]> : tensor<4x6xui32>
    return %c : tensor<4x6xui32>
  }
}
