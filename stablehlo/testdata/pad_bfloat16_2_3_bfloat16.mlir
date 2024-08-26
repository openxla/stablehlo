// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6x4xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3xbf16>, tensor<bf16>)
    %1 = call @expected() : () -> tensor<6x4xbf16>
    %2 = stablehlo.pad %0#0, %0#1, low = [1, 0], high = [2, 1], interior = [1, 0] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<6x4xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<6x4xbf16>, tensor<6x4xbf16>) -> ()
    return %2 : tensor<6x4xbf16>
  }
  func.func private @inputs() -> (tensor<2x3xbf16> {mhlo.layout_mode = "default"}, tensor<bf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.604890e-04, -2.956390e-04, 2.075200e-03], [-8.087160e-04, 1.773830e-04, -3.852840e-04]]> : tensor<2x3xbf16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    return %cst, %cst_0 : tensor<2x3xbf16>, tensor<bf16>
  }
  func.func private @expected() -> (tensor<6x4xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-3.604890e-04, -2.956390e-04, 2.075200e-03, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-8.087160e-04, 1.773830e-04, -3.852840e-04, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<6x4xbf16>
    return %cst : tensor<6x4xbf16>
  }
}
