// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @expected() : () -> tensor<2x3xui8>
    %1 = stablehlo.iota dim = 0 : tensor<2x3xui8>
    stablehlo.custom_call @check.expect_eq(%1, %0) {has_side_effect = true} : (tensor<2x3xui8>, tensor<2x3xui8>) -> ()
    return %1 : tensor<2x3xui8>
  }
  func.func private @expected() -> (tensor<2x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 0, 0], [1, 1, 1]]> : tensor<2x3xui8>
    return %c : tensor<2x3xui8>
  }
}
