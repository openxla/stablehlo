// RUN-DISABLED(no interpreter support) stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<14x15x0x33xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<14x15x0x17xcomplex<f64>>
    %1 = call @expected() : () -> tensor<14x15x0x33xf64>
    %2 = stablehlo.fft %0, type =  IRFFT, length = [33] : (tensor<14x15x0x17xcomplex<f64>>) -> tensor<14x15x0x33xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<14x15x0x33xf64>, tensor<14x15x0x33xf64>) -> ()
    return %2 : tensor<14x15x0x33xf64>
  }
  func.func private @inputs() -> (tensor<14x15x0x17xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<> : tensor<14x15x0x17xcomplex<f64>>
    return %cst : tensor<14x15x0x17xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<14x15x0x33xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<> : tensor<14x15x0x33xf64>
    return %cst : tensor<14x15x0x33xf64>
  }
}
