// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi32>, tensor<3x6xui8>)
    %1 = call @expected() : () -> tensor<4x6xi32>
    %2 = stablehlo.convert %0#0 : tensor<4x3xi32>
    %3 = stablehlo.convert %0#1 : (tensor<3x6xui8>) -> tensor<3x6xi32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xi32>, tensor<3x6xi32>) -> tensor<4x6xi32>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xi32>, tensor<4x6xi32>) -> ()
    return %4 : tensor<4x6xi32>
  }
  func.func private @inputs() -> (tensor<4x3xi32> {mhlo.layout_mode = "default"}, tensor<3x6xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-1, -2, -4], [0, -2, -2], [0, 0, 0], [3, -1, 4]]> : tensor<4x3xi32>
    %c_0 = stablehlo.constant dense<[[4, 5, 0, 1, 6, 1], [4, 1, 3, 5, 0, 4], [3, 2, 0, 4, 3, 1]]> : tensor<3x6xui8>
    return %c, %c_0 : tensor<4x3xi32>, tensor<3x6xui8>
  }
  func.func private @expected() -> (tensor<4x6xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-24, -15, -6, -27, -18, -13], [-14, -6, -6, -18, -6, -10], [0, 0, 0, 0, 0, 0], [20, 22, -3, 14, 30, 3]]> : tensor<4x6xi32>
    return %c : tensor<4x6xi32>
  }
}
