// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<7xf32>
    %1 = call @expected() : () -> tensor<3xf32>
    %2 = stablehlo.slice %0 [4:7] : (tensor<7xf32>) -> tensor<3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xf32>, tensor<3xf32>) -> ()
    return %2 : tensor<3xf32>
  }
  func.func private @inputs() -> (tensor<7xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-2.26640487, 1.58945656, -0.586266816, -5.04116297, 0.919672251, 1.11756611, 0.796692073]> : tensor<7xf32>
    return %cst : tensor<7xf32>
  }
  func.func private @expected() -> (tensor<3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0.919672251, 1.11756611, 0.796692073]> : tensor<3xf32>
    return %cst : tensor<3xf32>
  }
}
