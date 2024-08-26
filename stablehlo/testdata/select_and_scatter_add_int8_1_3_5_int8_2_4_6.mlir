// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x3x5xi8>, tensor<2x4x6xi8>)
    %1 = call @expected() : () -> tensor<2x4x6xi8>
    %c = stablehlo.constant dense<-128> : tensor<i8>
    %2 = stablehlo.pad %0#1, %c, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xi8>, tensor<i8>) -> tensor<2x4x6xi8>
    %c_0 = stablehlo.constant dense<0> : tensor<i8>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %c_0) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  SIGNED : (tensor<i8>, tensor<i8>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) : (tensor<2x4x6xi8>, tensor<1x3x5xi8>, tensor<i8>) -> tensor<2x4x6xi8>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xi8>) -> tensor<2x4x6xi8>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<2x4x6xi8>, tensor<2x4x6xi8>) -> ()
    return %4 : tensor<2x4x6xi8>
  }
  func.func private @inputs() -> (tensor<1x3x5xi8> {mhlo.layout_mode = "default"}, tensor<2x4x6xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[-5, 1, 2, -1, 0], [-1, 4, 4, -3, 0], [2, 0, -3, 0, -7]]]> : tensor<1x3x5xi8>
    %c_0 = stablehlo.constant dense<[[[-3, -6, 3, -4, 4, 0], [2, 2, -1, 2, 5, 2], [-1, 3, 5, 5, -1, 3], [0, 0, 1, 3, -1, -2]], [[1, 1, -3, -5, 0, 2], [-3, -4, -6, 2, 6, -1], [-3, 2, -1, 0, 2, 1], [4, 2, 2, -2, 1, 2]]]> : tensor<2x4x6xi8>
    return %c, %c_0 : tensor<1x3x5xi8>, tensor<2x4x6xi8>
  }
  func.func private @expected() -> (tensor<2x4x6xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 0, 3, 0, 0, 0], [-5, 0, 0, 0, 0, 0], [0, -1, 5, 0, 0, -7], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -4, 0], [0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]]]> : tensor<2x4x6xi8>
    return %c : tensor<2x4x6xi8>
  }
}
