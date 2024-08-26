// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi32>, tensor<3x6xbf16>)
    %1 = call @expected() : () -> tensor<4x6xbf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi32>) -> tensor<4x3xbf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xbf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    return %4 : tensor<4x6xbf16>
  }
  func.func private @inputs() -> (tensor<4x3xi32> {mhlo.layout_mode = "default"}, tensor<3x6xbf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, -2, 1], [-3, 7, -5], [2, 5, 1], [0, -4, 0]]> : tensor<4x3xi32>
    %cst = stablehlo.constant dense<[[1.664060e+00, -1.382810e+00, -3.812500e+00, 1.523440e+00, -8.867180e-01, 3.457030e-01], [7.656250e-01, -7.375000e+00, 5.718750e+00, 2.203130e+00, 7.343750e-01, -6.738280e-02], [6.875000e-01, -2.832030e-01, -5.531250e+00, 1.187500e+00, -8.359380e-01, -2.078130e+00]]> : tensor<3x6xbf16>
    return %c, %cst : tensor<4x3xi32>, tensor<3x6xbf16>
  }
  func.func private @expected() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-8.437500e-01, 1.443750e+01, -1.700000e+01, -3.218750e+00, -2.312500e+00, -1.945310e+00], [-3.062500e+00, -4.600000e+01, 7.900000e+01, 4.906250e+00, 1.200000e+01, 8.875000e+00], [7.843750e+00, -4.000000e+01, 1.543750e+01, 1.525000e+01, 1.062500e+00, -1.726560e+00], [-3.062500e+00, 2.950000e+01, -2.287500e+01, -8.812500e+00, -2.937500e+00, 2.695310e-01]]> : tensor<4x6xbf16>
    return %cst : tensor<4x6xbf16>
  }
}
