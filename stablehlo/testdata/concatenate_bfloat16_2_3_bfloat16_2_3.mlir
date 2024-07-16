// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3xbf16>, tensor<2x3xbf16>)
    %1 = call @expected() : () -> tensor<4x3xbf16>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 0 : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<4x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x3xbf16>, tensor<4x3xbf16>) -> ()
    return %2 : tensor<4x3xbf16>
  }
  func.func private @inputs() -> (tensor<2x3xbf16> {mhlo.layout_mode = "default"}, tensor<2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.375000e+00, -2.265630e+00, -3.554690e-01], [3.686520e-02, 3.640630e+00, 8.984370e-01]]> : tensor<2x3xbf16>
    %cst_0 = stablehlo.constant dense<[[-2.750000e+00, 2.203130e+00, -2.156250e+00], [2.906250e+00, 5.343750e+00, 4.437500e+00]]> : tensor<2x3xbf16>
    return %cst, %cst_0 : tensor<2x3xbf16>, tensor<2x3xbf16>
  }
  func.func private @expected() -> (tensor<4x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.375000e+00, -2.265630e+00, -3.554690e-01], [3.686520e-02, 3.640630e+00, 8.984370e-01], [-2.750000e+00, 2.203130e+00, -2.156250e+00], [2.906250e+00, 5.343750e+00, 4.437500e+00]]> : tensor<4x3xbf16>
    return %cst : tensor<4x3xbf16>
  }
}
