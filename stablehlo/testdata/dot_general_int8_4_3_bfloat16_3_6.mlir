// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi8>, tensor<3x6xbf16>)
    %1 = call @expected() : () -> tensor<4x6xbf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi8>) -> tensor<4x3xbf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xbf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    return %4 : tensor<4x6xbf16>
  }
  func.func private @inputs() -> (tensor<4x3xi8> {mhlo.layout_mode = "default"}, tensor<3x6xbf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-4, 0, 3], [-3, 1, -2], [1, -2, 4], [3, 0, 1]]> : tensor<4x3xi8>
    %cst = stablehlo.constant dense<[[-1.375000e+00, -2.406250e+00, -1.617190e+00, 8.945310e-01, 5.976560e-01, -1.117190e+00], [2.062500e+00, 2.468750e+00, 4.250000e+00, -5.703130e-01, -7.343750e-01, -5.937500e+00], [1.710940e+00, -5.031250e+00, -1.437500e+00, 2.421880e+00, 2.609380e+00, -2.089840e-01]]> : tensor<3x6xbf16>
    return %c, %cst : tensor<4x3xi8>, tensor<3x6xbf16>
  }
  func.func private @expected() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.062500e+01, -5.468750e+00, 2.156250e+00, 3.687500e+00, 5.437500e+00, 3.843750e+00], [2.765630e+00, 1.975000e+01, 1.200000e+01, -8.125000e+00, -7.750000e+00, -2.171880e+00], [1.343750e+00, -2.750000e+01, -1.587500e+01, 1.175000e+01, 1.250000e+01, 9.937500e+00], [-2.406250e+00, -1.225000e+01, -6.281250e+00, 5.093750e+00, 4.406250e+00, -3.562500e+00]]> : tensor<4x6xbf16>
    return %cst : tensor<4x6xbf16>
  }
}
