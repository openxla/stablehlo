// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xbf16>
    %1 = call @expected() : () -> tensor<3xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.multiply across dimensions = [0] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xbf16>, tensor<3xbf16>) -> ()
    return %2 : tensor<3xbf16>
  }
  func.func private @inputs() -> (tensor<2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.937500e+00, -2.412110e-01, -5.593750e+00], [-3.015630e+00, 9.187500e+00, -1.375000e+00]]> : tensor<2x3xbf16>
    return %cst : tensor<2x3xbf16>
  }
  func.func private @expected() -> (tensor<3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[1.487500e+01, -2.218750e+00, 7.687500e+00]> : tensor<3xbf16>
    return %cst : tensor<3xbf16>
  }
}
