// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xui32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<ui32>, tensor<2x3xui32>, tensor<ui32>)
    %1 = call @expected() : () -> tensor<2x3xui32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<ui32>) -> tensor<2x3xui32>
    %3 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<ui32>) -> tensor<2x3xui32>
    %4 = stablehlo.clamp %2, %0#1, %3 : tensor<2x3xui32>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<2x3xui32>, tensor<2x3xui32>) -> ()
    return %4 : tensor<2x3xui32>
  }
  func.func private @inputs() -> (tensor<ui32> {mhlo.layout_mode = "default"}, tensor<2x3xui32> {mhlo.layout_mode = "default"}, tensor<ui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 0, 2], [0, 0, 2]]> : tensor<2x3xui32>
    %c_0 = stablehlo.constant dense<2> : tensor<ui32>
    %c_1 = stablehlo.constant dense<0> : tensor<ui32>
    return %c_0, %c, %c_1 : tensor<ui32>, tensor<2x3xui32>, tensor<ui32>
  }
  func.func private @expected() -> (tensor<2x3xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<2x3xui32>
    return %c : tensor<2x3xui32>
  }
}
