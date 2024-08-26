// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x1x6xi16>, tensor<2x4x6xi16>)
    %1 = call @expected() : () -> tensor<2x4x6xi16>
    %c = stablehlo.constant dense<-32768> : tensor<i16>
    %2 = stablehlo.pad %0#1, %c, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xi16>, tensor<i16>) -> tensor<2x4x6xi16>
    %c_0 = stablehlo.constant dense<0> : tensor<i16>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %c_0) <{window_dimensions = array<i64: 1, 3, 1>, window_strides = array<i64: 1, 2, 1>}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  SIGNED : (tensor<i16>, tensor<i16>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) : (tensor<2x4x6xi16>, tensor<2x1x6xi16>, tensor<i16>) -> tensor<2x4x6xi16>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xi16>) -> tensor<2x4x6xi16>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<2x4x6xi16>, tensor<2x4x6xi16>) -> ()
    return %4 : tensor<2x4x6xi16>
  }
  func.func private @inputs() -> (tensor<2x1x6xi16> {mhlo.layout_mode = "default"}, tensor<2x4x6xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[4, 1, 0, -1, 2, 1]], [[2, 0, 1, -2, -4, 1]]]> : tensor<2x1x6xi16>
    %c_0 = stablehlo.constant dense<[[[0, -1, 0, -4, -1, 0], [-1, -1, 0, 5, -5, 1], [-2, 3, -2, 3, 0, -3], [-1, -2, 3, 3, 0, -2]], [[0, -5, -1, 0, 0, 2], [0, 3, -1, 0, 1, 4], [-1, 6, -4, 0, 0, 2], [-5, 0, -2, -7, 1, 2]]]> : tensor<2x4x6xi16>
    return %c, %c_0 : tensor<2x1x6xi16>, tensor<2x4x6xi16>
  }
  func.func private @expected() -> (tensor<2x4x6xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[4, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 1], [0, 1, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0]], [[2, 0, 1, -2, 0, 0], [0, 0, 0, 0, -4, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]]> : tensor<2x4x6xi16>
    return %c : tensor<2x4x6xi16>
  }
}
