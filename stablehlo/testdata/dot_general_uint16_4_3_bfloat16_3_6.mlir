// RUN-DISABLED(inaccurate) stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui16>, tensor<3x6xbf16>)
    %1 = call @expected() : () -> tensor<4x6xbf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui16>) -> tensor<4x3xbf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xbf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    stablehlo.custom_call @check.expect_almost_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    return %4 : tensor<4x6xbf16>
  }
  func.func private @inputs() -> (tensor<4x3xui16> {mhlo.layout_mode = "default"}, tensor<3x6xbf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 2, 3], [1, 2, 1], [4, 3, 4], [1, 4, 1]]> : tensor<4x3xui16>
    %cst = stablehlo.constant dense<[[-1.664060e+00, -6.937500e+00, 3.203130e-01, -1.765630e+00, 1.476560e+00, 1.562500e+00], [-2.765630e+00, 1.468750e+00, 3.476560e-01, 3.398440e-01, 2.218750e+00, 9.843750e-01], [1.531250e+00, -2.296880e+00, -5.781250e-01, -2.171880e+00, 5.562500e+00, -2.796880e+00]]> : tensor<3x6xbf16>
    return %c, %cst : tensor<4x3xui16>, tensor<3x6xbf16>
  }
  func.func private @expected() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.250000e+00, -1.787500e+01, -3.984380e-01, -9.375000e+00, 2.412500e+01, -3.296880e+00], [-5.656250e+00, -6.312500e+00, 4.375000e-01, -3.250000e+00, 1.150000e+01, 7.343750e-01], [-8.812500e+00, -3.250000e+01, 1.171880e-02, -1.475000e+01, 3.475000e+01, -1.984380e+00], [-1.118750e+01, -3.359380e+00, 1.132810e+00, -2.578130e+00, 1.593750e+01, 2.703130e+00]]> : tensor<4x6xbf16>
    return %cst : tensor<4x6xbf16>
  }
}
