// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xi64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi8>, tensor<3x6xi64>)
    %1 = call @expected() : () -> tensor<4x6xi64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi8>) -> tensor<4x3xi64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xi64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xi64>, tensor<3x6xi64>) -> tensor<4x6xi64>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xi64>, tensor<4x6xi64>) -> ()
    return %4 : tensor<4x6xi64>
  }
  func.func private @inputs() -> (tensor<4x3xi8> {mhlo.layout_mode = "default"}, tensor<3x6xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[7, 6, 0], [0, 1, -1], [2, 1, -4], [3, 0, 1]]> : tensor<4x3xi8>
    %c_0 = stablehlo.constant dense<[[1, 1, -3, 7, 0, 0], [0, 0, 5, -5, -1, 0], [0, 5, 0, 4, 4, 1]]> : tensor<3x6xi64>
    return %c, %c_0 : tensor<4x3xi8>, tensor<3x6xi64>
  }
  func.func private @expected() -> (tensor<4x6xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[7, 7, 9, 19, -6, 0], [0, -5, 5, -9, -5, -1], [2, -18, -1, -7, -17, -4], [3, 8, -9, 25, 4, 1]]> : tensor<4x6xi64>
    return %c : tensor<4x6xi64>
  }
}
