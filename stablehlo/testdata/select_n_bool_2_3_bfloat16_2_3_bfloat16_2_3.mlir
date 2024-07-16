// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<2x3xi1>, tensor<2x3xbf16>, tensor<2x3xbf16>)
    %1 = call @expected() : () -> tensor<2x3xbf16>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<2x3xi1>, tensor<2x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> ()
    return %2 : tensor<2x3xbf16>
  }
  func.func private @inputs() -> (tensor<2x3xi1> {mhlo.layout_mode = "default"}, tensor<2x3xbf16> {mhlo.layout_mode = "default"}, tensor<2x3xbf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<2x3xi1>
    %cst = stablehlo.constant dense<[[2.906250e+00, -2.843750e+00, 1.281250e+00], [-3.593750e+00, 2.406250e+00, -2.531250e+00]]> : tensor<2x3xbf16>
    %cst_0 = stablehlo.constant dense<[[5.875000e+00, -1.742190e+00, 3.234380e+00], [-4.718750e+00, 5.781250e-01, 4.750000e+00]]> : tensor<2x3xbf16>
    return %c, %cst, %cst_0 : tensor<2x3xi1>, tensor<2x3xbf16>, tensor<2x3xbf16>
  }
  func.func private @expected() -> (tensor<2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[5.875000e+00, -1.742190e+00, 3.234380e+00], [-4.718750e+00, 5.781250e-01, 4.750000e+00]]> : tensor<2x3xbf16>
    return %cst : tensor<2x3xbf16>
  }
}
