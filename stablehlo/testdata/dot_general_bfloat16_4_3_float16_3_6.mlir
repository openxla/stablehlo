// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xbf16>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xbf16>
    %2 = stablehlo.convert %0#0 : tensor<4x3xbf16>
    %3 = stablehlo.convert %0#1 : (tensor<3x6xf16>) -> tensor<3x6xbf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    return %4 : tensor<4x6xbf16>
  }
  func.func private @inputs() -> (tensor<4x3xbf16> {mhlo.layout_mode = "default"}, tensor<3x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.711910e-02, 6.367190e-01, -3.027340e-01], [3.203130e+00, 4.199220e-01, 3.062500e+00], [2.203130e+00, 1.031250e+00, 1.632810e+00], [4.003910e-01, 4.375000e+00, -2.171880e+00]]> : tensor<4x3xbf16>
    %cst_0 = stablehlo.constant dense<[[-6.925780e+00, -4.007810e+00, 2.437740e-01, -5.703130e+00, 5.039060e+00, -1.204100e+00], [-1.289060e+00, 4.794920e-01, -1.965820e+00, -6.714840e+00, -2.487180e-03, -5.101560e+00], [-2.066410e+00, 1.361330e+00, 3.738280e+00, 6.655270e-01, 4.648440e+00, -4.504390e-01]]> : tensor<3x6xf16>
    return %cst, %cst_0 : tensor<4x3xbf16>, tensor<3x6xf16>
  }
  func.func private @expected() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.308590e-01, 8.300780e-02, -2.390630e+00, -4.218750e+00, -1.648440e+00, -3.046880e+00], [-2.912500e+01, -8.437500e+00, 1.137500e+01, -1.900000e+01, 3.037500e+01, -7.375000e+00], [-2.000000e+01, -6.093750e+00, 4.593750e+00, -1.837500e+01, 1.862500e+01, -8.625000e+00], [-3.937500e+00, -2.453130e+00, -1.662500e+01, -3.300000e+01, -8.125000e+00, -2.175000e+01]]> : tensor<4x6xbf16>
    return %cst : tensor<4x6xbf16>
  }
}
