// RUN-DISABLED(inaccurate) stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui32>, tensor<3x6xbf16>)
    %1 = call @expected() : () -> tensor<4x6xbf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui32>) -> tensor<4x3xbf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xbf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    stablehlo.custom_call @check.expect_almost_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    return %4 : tensor<4x6xbf16>
  }
  func.func private @inputs() -> (tensor<4x3xui32> {mhlo.layout_mode = "default"}, tensor<3x6xbf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[7, 2, 1], [2, 3, 6], [0, 0, 0], [1, 3, 0]]> : tensor<4x3xui32>
    %cst = stablehlo.constant dense<[[4.082030e-01, 6.625000e+00, 4.042970e-01, 1.867190e+00, 1.664060e+00, 2.203130e+00], [-2.187500e+00, -2.171880e+00, 3.781250e+00, -3.234380e+00, -1.890630e+00, -2.015630e+00], [-3.921880e+00, -2.437500e+00, -4.468750e+00, -4.156250e+00, -2.138670e-01, 2.460940e-01]]> : tensor<3x6xbf16>
    return %c, %cst : tensor<4x3xui32>, tensor<3x6xbf16>
  }
  func.func private @expected() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-5.437500e+00, 3.950000e+01, 5.937500e+00, 2.437500e+00, 7.656250e+00, 1.162500e+01], [-2.925000e+01, -7.875000e+00, -1.468750e+01, -3.087500e+01, -3.625000e+00, -1.640630e-01], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-6.156250e+00, 1.093750e-01, 1.175000e+01, -7.843750e+00, -4.000000e+00, -3.843750e+00]]> : tensor<4x6xbf16>
    return %cst : tensor<4x6xbf16>
  }
}
