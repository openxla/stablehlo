// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xui32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x3x5xui32>, tensor<2x4x6xui32>)
    %1 = call @expected() : () -> tensor<2x4x6xui32>
    %c = stablehlo.constant dense<0> : tensor<ui32>
    %2 = stablehlo.pad %0#1, %c, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xui32>, tensor<ui32>) -> tensor<2x4x6xui32>
    %c_0 = stablehlo.constant dense<0> : tensor<ui32>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %c_0) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  UNSIGNED : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui32>
      stablehlo.return %5 : tensor<ui32>
    }) : (tensor<2x4x6xui32>, tensor<1x3x5xui32>, tensor<ui32>) -> tensor<2x4x6xui32>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xui32>) -> tensor<2x4x6xui32>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<2x4x6xui32>, tensor<2x4x6xui32>) -> ()
    return %4 : tensor<2x4x6xui32>
  }
  func.func private @inputs() -> (tensor<1x3x5xui32> {mhlo.layout_mode = "default"}, tensor<2x4x6xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 2, 0, 1, 1], [0, 2, 3, 1, 5], [3, 1, 0, 2, 6]]]> : tensor<1x3x5xui32>
    %c_0 = stablehlo.constant dense<[[[3, 1, 3, 2, 7, 1], [0, 2, 1, 0, 1, 1], [1, 5, 4, 3, 2, 5], [1, 0, 1, 0, 5, 2]], [[6, 2, 3, 0, 2, 0], [2, 0, 0, 0, 1, 0], [2, 3, 0, 3, 4, 1], [0, 2, 0, 2, 6, 1]]]> : tensor<2x4x6xui32>
    return %c, %c_0 : tensor<1x3x5xui32>, tensor<2x4x6xui32>
  }
  func.func private @expected() -> (tensor<2x4x6xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 0, 2, 0, 2, 0], [0, 0, 0, 0, 0, 0], [0, 6, 3, 0, 0, 5], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 8, 0]]]> : tensor<2x4x6xui32>
    return %c : tensor<2x4x6xui32>
  }
}
