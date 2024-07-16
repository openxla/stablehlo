// RUN-DISABLED(no interpreter support): stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x1xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x4xf16>, tensor<4x1xf16>)
    %1 = call @expected() : () -> tensor<4x1xf16>
    %2 = "stablehlo.triangular_solve"(%0#0, %0#1) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<4x4xf16>, tensor<4x1xf16>) -> tensor<4x1xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x1xf16>, tensor<4x1xf16>) -> ()
    return %2 : tensor<4x1xf16>
  }
  func.func private @inputs() -> (tensor<4x4xf16> {mhlo.layout_mode = "default"}, tensor<4x1xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.134770e+00, -1.165040e+00, -1.280270e+00, 3.953130e+00], [-4.468750e+00, 2.023440e+00, -2.595210e-01, 2.658200e+00], [7.519530e-02, -1.421880e+00, -1.767580e+00, 4.679690e+00], [4.449220e+00, -1.000980e+00, -9.109370e+00, -2.769530e+00]]> : tensor<4x4xf16>
    %cst_0 = stablehlo.constant dense<[[6.445310e+00], [3.988280e+00], [-1.346440e-01], [-8.625000e+00]]> : tensor<4x1xf16>
    return %cst, %cst_0 : tensor<4x4xf16>, tensor<4x1xf16>
  }
  func.func private @expected() -> (tensor<4x1xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.355000e+02], [3.734380e+01], [4.021880e+01], [-8.625000e+00]]> : tensor<4x1xf16>
    return %cst : tensor<4x1xf16>
  }
}
