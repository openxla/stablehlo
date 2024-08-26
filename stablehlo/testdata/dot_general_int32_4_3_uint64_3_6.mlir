// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xui64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi32>, tensor<3x6xui64>)
    %1 = call @expected() : () -> tensor<4x6xui64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi32>) -> tensor<4x3xui64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xui64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xui64>, tensor<3x6xui64>) -> tensor<4x6xui64>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xui64>, tensor<4x6xui64>) -> ()
    return %4 : tensor<4x6xui64>
  }
  func.func private @inputs() -> (tensor<4x3xi32> {mhlo.layout_mode = "default"}, tensor<3x6xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 5, 0], [3, 1, -5], [5, -2, -1], [0, -7, 1]]> : tensor<4x3xi32>
    %c_0 = stablehlo.constant dense<[[1, 2, 0, 1, 1, 2], [3, 3, 3, 0, 5, 4], [6, 0, 2, 1, 6, 5]]> : tensor<3x6xui64>
    return %c, %c_0 : tensor<4x3xi32>, tensor<3x6xui64>
  }
  func.func private @expected() -> (tensor<4x6xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[18, 21, 15, 3, 28, 26], [18446744073709551592, 9, 18446744073709551609, 18446744073709551614, 18446744073709551594, 18446744073709551601], [18446744073709551609, 4, 18446744073709551608, 4, 18446744073709551605, 18446744073709551613], [18446744073709551601, 18446744073709551595, 18446744073709551597, 1, 18446744073709551587, 18446744073709551593]]> : tensor<4x6xui64>
    return %c : tensor<4x6xui64>
  }
}
