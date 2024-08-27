// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3xf64>
    %1 = call @expected() : () -> tensor<1xf64>
    %2 = stablehlo.slice %0 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1xf64>, tensor<1xf64>) -> ()
    return %2 : tensor<1xf64>
  }
  func.func private @inputs() -> (tensor<3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-0.51773568596824759, 1.8601306351761653, 3.3756521433494546]> : tensor<3xf64>
    return %cst : tensor<3xf64>
  }
  func.func private @expected() -> (tensor<1xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<1.8601306351761653> : tensor<1xf64>
    return %cst : tensor<1xf64>
  }
}
