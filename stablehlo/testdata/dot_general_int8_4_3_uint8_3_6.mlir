// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi8>, tensor<3x6xui8>)
    %1 = call @expected() : () -> tensor<4x6xi8>
    %2 = stablehlo.convert %0#0 : tensor<4x3xi8>
    %3 = stablehlo.convert %0#1 : (tensor<3x6xui8>) -> tensor<3x6xi8>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xi8>, tensor<3x6xi8>) -> tensor<4x6xi8>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xi8>, tensor<4x6xi8>) -> ()
    return %4 : tensor<4x6xi8>
  }
  func.func private @inputs() -> (tensor<4x3xi8> {mhlo.layout_mode = "default"}, tensor<3x6xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 9, 0], [3, -1, 2], [-4, -5, -4], [-1, 2, -1]]> : tensor<4x3xi8>
    %c_0 = stablehlo.constant dense<[[1, 0, 4, 2, 0, 3], [0, 6, 6, 3, 0, 2], [0, 2, 0, 1, 1, 0]]> : tensor<3x6xui8>
    return %c, %c_0 : tensor<4x3xi8>, tensor<3x6xui8>
  }
  func.func private @expected() -> (tensor<4x6xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 54, 54, 27, 0, 18], [3, -2, 6, 5, 2, 7], [-4, -38, -46, -27, -4, -22], [-1, 10, 8, 3, -1, 1]]> : tensor<4x6xi8>
    return %c : tensor<4x6xi8>
  }
}
