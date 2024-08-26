// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xf64>
    %1 = call @expected() : () -> tensor<3xf64>
    %cst = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.maximum across dimensions = [0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xf64>, tensor<3xf64>) -> ()
    return %2 : tensor<3xf64>
  }
  func.func private @inputs() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.26309286187879577, -1.9087842265701214, -0.64194290673004495], [2.3193875620984707, -5.559375664459191, 3.3056404880737893]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
  func.func private @expected() -> (tensor<3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[2.3193875620984707, -1.9087842265701214, 3.3056404880737893]> : tensor<3xf64>
    return %cst : tensor<3xf64>
  }
}
