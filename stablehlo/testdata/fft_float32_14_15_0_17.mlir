// RUN-DISABLED(no interpreter support) stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<14x15x0x9xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<14x15x0x17xf32>
    %1 = call @expected() : () -> tensor<14x15x0x9xcomplex<f32>>
    %2 = stablehlo.fft %0, type =  RFFT, length = [17] : (tensor<14x15x0x17xf32>) -> tensor<14x15x0x9xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<14x15x0x9xcomplex<f32>>, tensor<14x15x0x9xcomplex<f32>>) -> ()
    return %2 : tensor<14x15x0x9xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<14x15x0x17xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<> : tensor<14x15x0x17xf32>
    return %cst : tensor<14x15x0x17xf32>
  }
  func.func private @expected() -> (tensor<14x15x0x9xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<> : tensor<14x15x0x9xcomplex<f32>>
    return %cst : tensor<14x15x0x9xcomplex<f32>>
  }
}
