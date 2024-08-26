// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<ui8>, tensor<2x3xui8>, tensor<ui8>)
    %1 = call @expected() : () -> tensor<2x3xui8>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<ui8>) -> tensor<2x3xui8>
    %3 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<ui8>) -> tensor<2x3xui8>
    %4 = stablehlo.clamp %2, %0#1, %3 : tensor<2x3xui8>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<2x3xui8>, tensor<2x3xui8>) -> ()
    return %4 : tensor<2x3xui8>
  }
  func.func private @inputs() -> (tensor<ui8> {mhlo.layout_mode = "default"}, tensor<2x3xui8> {mhlo.layout_mode = "default"}, tensor<ui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, 1, 1], [1, 4, 3]]> : tensor<2x3xui8>
    %c_0 = stablehlo.constant dense<3> : tensor<ui8>
    %c_1 = stablehlo.constant dense<2> : tensor<ui8>
    return %c_0, %c, %c_1 : tensor<ui8>, tensor<2x3xui8>, tensor<ui8>
  }
  func.func private @expected() -> (tensor<2x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<2> : tensor<2x3xui8>
    return %c : tensor<2x3xui8>
  }
}
