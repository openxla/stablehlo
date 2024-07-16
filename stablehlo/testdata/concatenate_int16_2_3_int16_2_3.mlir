// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x3xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3xi16>, tensor<2x3xi16>)
    %1 = call @expected() : () -> tensor<4x3xi16>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 0 : (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<4x3xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x3xi16>, tensor<4x3xi16>) -> ()
    return %2 : tensor<4x3xi16>
  }
  func.func private @inputs() -> (tensor<2x3xi16> {mhlo.layout_mode = "default"}, tensor<2x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, -2, -1], [4, 0, 0]]> : tensor<2x3xi16>
    %c_0 = stablehlo.constant dense<[[3, 2, -2], [0, -2, -4]]> : tensor<2x3xi16>
    return %c, %c_0 : tensor<2x3xi16>, tensor<2x3xi16>
  }
  func.func private @expected() -> (tensor<4x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, -2, -1], [4, 0, 0], [3, 2, -2], [0, -2, -4]]> : tensor<4x3xi16>
    return %c : tensor<4x3xi16>
  }
}
