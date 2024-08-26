// RUN-DISABLED(inaccurate) stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui64>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui64>) -> tensor<4x3xf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    stablehlo.custom_call @check.expect_almost_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    return %4 : tensor<4x6xf16>
  }
  func.func private @inputs() -> (tensor<4x3xui64> {mhlo.layout_mode = "default"}, tensor<3x6xf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 2, 1], [4, 0, 6], [0, 2, 3], [0, 0, 0]]> : tensor<4x3xui64>
    %cst = stablehlo.constant dense<[[1.230470e+00, 1.026370e+00, 4.164060e+00, 5.234380e+00, -1.606450e+00, 1.020510e+00], [1.407230e+00, -4.345700e-01, 3.222660e+00, -7.607420e-01, -4.353030e-01, -2.222660e+00], [4.625000e+00, -1.030270e+00, 3.173830e+00, 3.960940e+00, 4.909670e-01, -6.254880e-01]]> : tensor<3x6xf16>
    return %c, %cst : tensor<4x3xui64>, tensor<3x6xf16>
  }
  func.func private @expected() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[8.671870e+00, -8.730460e-01, 1.378130e+01, 7.671880e+00, -1.986330e+00, -4.050780e+00], [3.268750e+01, -2.076170e+00, 3.568750e+01, 4.468750e+01, -3.480470e+00, 3.291020e-01], [1.668750e+01, -3.960940e+00, 1.596880e+01, 1.035940e+01, 6.025390e-01, -6.320310e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<4x6xf16>
    return %cst : tensor<4x6xf16>
  }
}
