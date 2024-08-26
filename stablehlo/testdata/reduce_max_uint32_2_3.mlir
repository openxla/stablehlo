// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xui32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xui32>
    %1 = call @expected() : () -> tensor<3xui32>
    %c = stablehlo.constant dense<0> : tensor<ui32>
    %2 = stablehlo.reduce(%0 init: %c) applies stablehlo.maximum across dimensions = [0] : (tensor<2x3xui32>, tensor<ui32>) -> tensor<3xui32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3xui32>, tensor<3xui32>) -> ()
    return %2 : tensor<3xui32>
  }
  func.func private @inputs() -> (tensor<2x3xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 5, 2], [3, 0, 4]]> : tensor<2x3xui32>
    return %c : tensor<2x3xui32>
  }
  func.func private @expected() -> (tensor<3xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 5, 4]> : tensor<3xui32>
    return %c : tensor<3xui32>
  }
}
