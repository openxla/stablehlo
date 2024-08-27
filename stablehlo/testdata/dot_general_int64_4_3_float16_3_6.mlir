// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi64>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi64>) -> tensor<4x3xf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    return %4 : tensor<4x6xf16>
  }
  func.func private @inputs() -> (tensor<4x3xi64> {mhlo.layout_mode = "default"}, tensor<3x6xf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, -2, -7], [-1, -2, 5], [1, 4, 1], [-6, 2, -1]]> : tensor<4x3xi64>
    %cst = stablehlo.constant dense<[[1.152340e+00, -4.539060e+00, -6.997070e-01, 2.373050e+00, -2.224610e+00, 1.533200e+00], [-3.447270e+00, 8.027340e-01, -2.769530e+00, -8.725580e-01, -4.027340e+00, -3.121090e+00], [1.754880e+00, -2.937500e+00, 2.408200e+00, 3.320310e+00, 2.689450e+00, 4.411620e-01]]> : tensor<3x6xf16>
    return %c, %cst : tensor<4x3xi64>, tensor<3x6xf16>
  }
  func.func private @expected() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-5.390630e+00, 1.895310e+01, -1.132030e+01, -2.150000e+01, -1.077340e+01, 3.154300e+00], [1.451560e+01, -1.175000e+01, 1.828130e+01, 1.597660e+01, 2.371880e+01, 6.914060e+00], [-1.088280e+01, -4.265630e+00, -9.367180e+00, 2.203130e+00, -1.564060e+01, -1.050780e+01], [-1.556250e+01, 3.178130e+01, -3.750000e+00, -1.929690e+01, 2.603520e+00, -1.588280e+01]]> : tensor<4x6xf16>
    return %cst : tensor<4x6xf16>
  }
}
