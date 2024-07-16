// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<f64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<f64>
    %1 = call @expected() : () -> tensor<f64>
    %2 = stablehlo.reduce_precision %0, format = e5m10 : tensor<f64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<f64>, tensor<f64>) -> ()
    return %2 : tensor<f64>
  }
  func.func private @inputs() -> (tensor<f64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<1.2006864362520839> : tensor<f64>
    return %cst : tensor<f64>
  }
  func.func private @expected() -> (tensor<f64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<1.201171875> : tensor<f64>
    return %cst : tensor<f64>
  }
}
