// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xi1> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<20x20xi1>, tensor<20x20xi1>)
    %1 = call @expected() : () -> tensor<20x20xi1>
    %2 = stablehlo.maximum %0#0, %0#1 : tensor<20x20xi1>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<20x20xi1>, tensor<20x20xi1>) -> ()
    return %2 : tensor<20x20xi1>
  }
  func.func private @inputs() -> (tensor<20x20xi1> {mhlo.layout_mode = "default"}, tensor<20x20xi1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<20x20xi1>
    %c_0 = stablehlo.constant dense<true> : tensor<20x20xi1>
    return %c, %c_0 : tensor<20x20xi1>, tensor<20x20xi1>
  }
  func.func private @expected() -> (tensor<20x20xi1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<20x20xi1>
    return %c : tensor<20x20xi1>
  }
}
