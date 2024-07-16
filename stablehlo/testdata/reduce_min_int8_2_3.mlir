// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xi8>
    %1 = call @expected() : () -> tensor<3xi8>
    %c = stablehlo.constant dense<127> : tensor<i8>
    %2 = stablehlo.reduce(%0 init: %c) applies stablehlo.minimum across dimensions = [0] : (tensor<2x3xi8>, tensor<i8>) -> tensor<3xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3xi8>, tensor<3xi8>) -> ()
    return %2 : tensor<3xi8>
  }
  func.func private @inputs() -> (tensor<2x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 0, 0], [-3, -3, -5]]> : tensor<2x3xi8>
    return %c : tensor<2x3xi8>
  }
  func.func private @expected() -> (tensor<3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[-3, -3, -5]> : tensor<3xi8>
    return %c : tensor<3xi8>
  }
}
