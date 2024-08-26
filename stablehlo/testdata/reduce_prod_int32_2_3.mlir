// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xi32>
    %1 = call @expected() : () -> tensor<3xi32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %2 = stablehlo.reduce(%0 init: %c) applies stablehlo.multiply across dimensions = [0] : (tensor<2x3xi32>, tensor<i32>) -> tensor<3xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3xi32>, tensor<3xi32>) -> ()
    return %2 : tensor<3xi32>
  }
  func.func private @inputs() -> (tensor<2x3xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-5, 3, 2], [1, -4, 3]]> : tensor<2x3xi32>
    return %c : tensor<2x3xi32>
  }
  func.func private @expected() -> (tensor<3xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[-5, -12, 6]> : tensor<3xi32>
    return %c : tensor<3xi32>
  }
}
