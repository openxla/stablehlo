// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x3x5xui16>, tensor<2x4x6xui16>)
    %1 = call @expected() : () -> tensor<2x4x6xui16>
    %c = stablehlo.constant dense<0> : tensor<ui16>
    %2 = stablehlo.pad %0#1, %c, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xui16>, tensor<ui16>) -> tensor<2x4x6xui16>
    %c_0 = stablehlo.constant dense<0> : tensor<ui16>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %c_0) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  UNSIGNED : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) : (tensor<2x4x6xui16>, tensor<1x3x5xui16>, tensor<ui16>) -> tensor<2x4x6xui16>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xui16>) -> tensor<2x4x6xui16>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<2x4x6xui16>, tensor<2x4x6xui16>) -> ()
    return %4 : tensor<2x4x6xui16>
  }
  func.func private @inputs() -> (tensor<1x3x5xui16> {mhlo.layout_mode = "default"}, tensor<2x4x6xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[2, 0, 5, 3, 0], [2, 0, 0, 3, 1], [4, 0, 1, 1, 0]]]> : tensor<1x3x5xui16>
    %c_0 = stablehlo.constant dense<[[[2, 3, 0, 1, 1, 2], [3, 0, 7, 3, 0, 7], [2, 3, 2, 1, 0, 3], [1, 0, 4, 1, 2, 5]], [[0, 0, 0, 2, 7, 3], [4, 3, 1, 0, 0, 0], [2, 4, 1, 3, 3, 0], [0, 3, 1, 2, 2, 3]]]> : tensor<2x4x6xui16>
    return %c, %c_0 : tensor<1x3x5xui16>, tensor<2x4x6xui16>
  }
  func.func private @expected() -> (tensor<2x4x6xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 0, 0, 0, 0, 0], [0, 0, 5, 3, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], [[0, 0, 0, 0, 3, 0], [4, 0, 0, 0, 0, 0], [0, 4, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]]]> : tensor<2x4x6xui16>
    return %c : tensor<2x4x6xui16>
  }
}
