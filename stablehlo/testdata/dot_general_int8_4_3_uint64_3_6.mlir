// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xui64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi8>, tensor<3x6xui64>)
    %1 = call @expected() : () -> tensor<4x6xui64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi8>) -> tensor<4x3xui64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xui64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xui64>, tensor<3x6xui64>) -> tensor<4x6xui64>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xui64>, tensor<4x6xui64>) -> ()
    return %4 : tensor<4x6xui64>
  }
  func.func private @inputs() -> (tensor<4x3xi8> {mhlo.layout_mode = "default"}, tensor<3x6xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, -1, -4], [-3, -1, -1], [1, 0, 0], [1, 4, 5]]> : tensor<4x3xi8>
    %c_0 = stablehlo.constant dense<[[0, 1, 1, 5, 1, 0], [3, 2, 2, 1, 0, 2], [2, 2, 0, 3, 1, 3]]> : tensor<3x6xui64>
    return %c, %c_0 : tensor<4x3xi8>, tensor<3x6xui64>
  }
  func.func private @expected() -> (tensor<4x6xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[18446744073709551605, 18446744073709551610, 2, 7, 0, 18446744073709551602], [18446744073709551611, 18446744073709551609, 18446744073709551611, 18446744073709551597, 18446744073709551612, 18446744073709551611], [0, 1, 1, 5, 1, 0], [22, 19, 9, 24, 6, 23]]> : tensor<4x6xui64>
    return %c : tensor<4x6xui64>
  }
}
