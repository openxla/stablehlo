// RUN-DISABLED(no interpreter support) stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2xui64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<ui64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2xui64>
    %1:2 = call @expected() : () -> (tensor<2xui64>, tensor<ui64>)
    %output_state, %output = stablehlo.rng_bit_generator %0, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
    stablehlo.custom_call @check.expect_eq(%output_state, %1#0) {has_side_effect = true} : (tensor<2xui64>, tensor<2xui64>) -> ()
    stablehlo.custom_call @check.expect_eq(%output, %1#1) {has_side_effect = true} : (tensor<ui64>, tensor<ui64>) -> ()
    return %output_state, %output : tensor<2xui64>, tensor<ui64>
  }
  func.func private @inputs() -> (tensor<2xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<3> : tensor<2xui64>
    return %c : tensor<2xui64>
  }
  func.func private @expected() -> (tensor<2xui64> {mhlo.layout_mode = "default"}, tensor<ui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 4]> : tensor<2xui64>
    %c_0 = stablehlo.constant dense<3349939604322698703> : tensor<ui64>
    return %c, %c_0 : tensor<2xui64>, tensor<ui64>
  }
}
