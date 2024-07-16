// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xi1> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x3x5xi1>, tensor<2x4x6xi1>)
    %1 = call @expected() : () -> tensor<2x4x6xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %2 = stablehlo.pad %0#1, %c, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xi1>, tensor<i1>) -> tensor<2x4x6xi1>
    %c_0 = stablehlo.constant dense<false> : tensor<i1>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %c_0) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg0: tensor<i1>, %arg1: tensor<i1>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<i1>, %arg1: tensor<i1>):
      %5 = stablehlo.or %arg0, %arg1 : tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }) : (tensor<2x4x6xi1>, tensor<1x3x5xi1>, tensor<i1>) -> tensor<2x4x6xi1>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xi1>) -> tensor<2x4x6xi1>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<2x4x6xi1>, tensor<2x4x6xi1>) -> ()
    return %4 : tensor<2x4x6xi1>
  }
  func.func private @inputs() -> (tensor<1x3x5xi1> {mhlo.layout_mode = "default"}, tensor<2x4x6xi1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<1x3x5xi1>
    %c_0 = stablehlo.constant dense<true> : tensor<2x4x6xi1>
    return %c, %c_0 : tensor<1x3x5xi1>, tensor<2x4x6xi1>
  }
  func.func private @expected() -> (tensor<2x4x6xi1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[true, true, true, true, true, false], [true, true, true, true, true, false], [true, true, true, true, true, false], [false, false, false, false, false, false]], [[false, false, false, false, false, false], [false, false, false, false, false, false], [false, false, false, false, false, false], [false, false, false, false, false, false]]]> : tensor<2x4x6xi1>
    return %c : tensor<2x4x6xi1>
  }
}
