// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x2xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<2x2xui16>
    %0 = call @expected() : () -> tensor<2x2xui16>
    %c_0 = stablehlo.constant dense<0> : tensor<ui16>
    %1 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui16>) -> tensor<2x2xui16>
    %2 = stablehlo.compare  EQ, %c, %1,  UNSIGNED : (tensor<2x2xui16>, tensor<2x2xui16>) -> tensor<2x2xi1>
    %c_1 = stablehlo.constant dense<0> : tensor<ui16>
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui16>) -> tensor<2x2xui16>
    %c_2 = stablehlo.constant dense<1> : tensor<ui16>
    %4 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui16>) -> tensor<2x2xui16>
    %5 = stablehlo.select %2, %3, %4 : tensor<2x2xi1>, tensor<2x2xui16>
    stablehlo.custom_call @check.expect_eq(%5, %0) {has_side_effect = true} : (tensor<2x2xui16>, tensor<2x2xui16>) -> ()
    return %5 : tensor<2x2xui16>
  }
  func.func private @expected() -> (tensor<2x2xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<2x2xui16>
    return %c : tensor<2x2xui16>
  }
}
