// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3x4xbf16>
    %1 = call @expected() : () -> tensor<3x4xbf16>
    stablehlo.custom_call @check.expect_close(%0, %1) {has_side_effect = true} : (tensor<3x4xbf16>, tensor<3x4xbf16>) -> ()
    return %0 : tensor<3x4xbf16>
  }
  func.func private @inputs() -> (tensor<3x4xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.890630e+00, -5.000000e+00, -3.968750e+00, -1.742190e+00], [6.750000e+00, -1.515630e+00, 5.125000e+00, 5.062500e+00], [-2.328130e+00, -3.859380e+00, -2.437500e+00, -3.609380e+00]]> : tensor<3x4xbf16>
    return %cst : tensor<3x4xbf16>
  }
  func.func private @expected() -> (tensor<3x4xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.890630e+00, -5.000000e+00, -3.968750e+00, -1.742190e+00], [6.750000e+00, -1.515630e+00, 5.125000e+00, 5.062500e+00], [-2.328130e+00, -3.859380e+00, -2.437500e+00, -3.609380e+00]]> : tensor<3x4xbf16>
    return %cst : tensor<3x4xbf16>
  }
}
