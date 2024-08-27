// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1xi1> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3xi1>
    %1 = call @expected() : () -> tensor<1xi1>
    %2 = stablehlo.slice %0 [1:2] : (tensor<3xi1>) -> tensor<1xi1>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1xi1>, tensor<1xi1>) -> ()
    return %2 : tensor<1xi1>
  }
  func.func private @inputs() -> (tensor<3xi1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<3xi1>
    return %c : tensor<3xi1>
  }
  func.func private @expected() -> (tensor<1xi1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<1xi1>
    return %c : tensor<1xi1>
  }
}
