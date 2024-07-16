// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi16>, tensor<3x6xui16>)
    %1 = call @expected() : () -> tensor<4x6xi16>
    %2 = stablehlo.convert %0#0 : tensor<4x3xi16>
    %3 = stablehlo.convert %0#1 : (tensor<3x6xui16>) -> tensor<3x6xi16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xi16>, tensor<3x6xi16>) -> tensor<4x6xi16>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xi16>, tensor<4x6xi16>) -> ()
    return %4 : tensor<4x6xi16>
  }
  func.func private @inputs() -> (tensor<4x3xi16> {mhlo.layout_mode = "default"}, tensor<3x6xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 0, 0], [2, -1, -4], [2, 0, 0], [-1, -3, 0]]> : tensor<4x3xi16>
    %c_0 = stablehlo.constant dense<[[0, 2, 3, 2, 4, 1], [5, 1, 0, 0, 4, 1], [2, 0, 2, 7, 1, 2]]> : tensor<3x6xui16>
    return %c, %c_0 : tensor<4x3xi16>, tensor<3x6xui16>
  }
  func.func private @expected() -> (tensor<4x6xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 0, 0, 0, 0, 0], [-13, 3, -2, -24, 0, -7], [0, 4, 6, 4, 8, 2], [-15, -5, -3, -2, -16, -4]]> : tensor<4x6xi16>
    return %c : tensor<4x6xi16>
  }
}
