// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<f16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<f16>
    %1 = call @expected() : () -> tensor<f16>
    %2 = stablehlo.reduce_precision %0, format = e11m52 : tensor<f16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<f16>, tensor<f16>) -> ()
    return %2 : tensor<f16>
  }
  func.func private @inputs() -> (tensor<f16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<5.087890e-01> : tensor<f16>
    return %cst : tensor<f16>
  }
  func.func private @expected() -> (tensor<f16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<5.087890e-01> : tensor<f16>
    return %cst : tensor<f16>
  }
}
