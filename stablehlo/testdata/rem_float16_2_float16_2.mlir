// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2xf16>, tensor<2xf16>)
    %1 = call @expected() : () -> tensor<2xf16>
    %2 = stablehlo.remainder %0#0, %0#1 : tensor<2xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2xf16>, tensor<2xf16>) -> ()
    return %2 : tensor<2xf16>
  }
  func.func private @inputs() -> (tensor<2xf16> {mhlo.layout_mode = "default"}, tensor<2xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[9.233390e-01, 1.684570e+00]> : tensor<2xf16>
    %cst_0 = stablehlo.constant dense<[5.457030e+00, -4.406250e+00]> : tensor<2xf16>
    return %cst, %cst_0 : tensor<2xf16>, tensor<2xf16>
  }
  func.func private @expected() -> (tensor<2xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[9.233390e-01, 1.684570e+00]> : tensor<2xf16>
    return %cst : tensor<2xf16>
  }
}
