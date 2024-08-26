// RUN-DISABLED(no interpreter support): stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x1xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x4xbf16>, tensor<4x1xbf16>)
    %1 = call @expected() : () -> tensor<4x1xbf16>
    %2 = "stablehlo.triangular_solve"(%0#0, %0#1) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<4x4xbf16>, tensor<4x1xbf16>) -> tensor<4x1xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x1xbf16>, tensor<4x1xbf16>) -> ()
    return %2 : tensor<4x1xbf16>
  }
  func.func private @inputs() -> (tensor<4x4xbf16> {mhlo.layout_mode = "default"}, tensor<4x1xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.515630e+00, -1.176760e-01, 1.195310e+00, 2.265630e+00], [-1.982420e-01, 2.734380e+00, 4.000000e+00, 2.734380e+00], [-2.812500e+00, -2.531250e+00, -3.027340e-01, -1.750000e+00], [1.023440e+00, 2.250000e+00, -7.265630e-01, 3.796880e+00]]> : tensor<4x4xbf16>
    %cst_0 = stablehlo.constant dense<[[4.437500e+00], [-4.042970e-01], [-4.375000e+00], [1.398440e+00]]> : tensor<4x1xbf16>
    return %cst, %cst_0 : tensor<4x4xbf16>, tensor<4x1xbf16>
  }
  func.func private @expected() -> (tensor<4x1xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.984380e+00], [3.468750e+00], [-1.929690e+00], [1.398440e+00]]> : tensor<4x1xbf16>
    return %cst : tensor<4x1xbf16>
  }
}
