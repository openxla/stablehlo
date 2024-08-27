// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi64>, tensor<3x6xbf16>)
    %1 = call @expected() : () -> tensor<4x6xbf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi64>) -> tensor<4x3xbf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xbf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    return %4 : tensor<4x6xbf16>
  }
  func.func private @inputs() -> (tensor<4x3xi64> {mhlo.layout_mode = "default"}, tensor<3x6xbf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 9, 2], [0, -4, 0], [-1, -2, 0], [-3, 2, -3]]> : tensor<4x3xi64>
    %cst = stablehlo.constant dense<[[-2.812500e+00, -4.785160e-01, -6.531250e+00, -3.531250e+00, -2.500000e+00, 7.734380e-01], [-2.546880e+00, 4.343750e+00, -2.140630e+00, -8.515620e-01, 2.484380e+00, 2.890630e+00], [1.882810e+00, -1.328130e+00, 1.750000e+00, -5.625000e-01, -1.078130e+00, 6.031250e+00]]> : tensor<3x6xbf16>
    return %c, %cst : tensor<4x3xi64>, tensor<3x6xbf16>
  }
  func.func private @expected() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.475000e+01, 3.550000e+01, -2.887500e+01, -1.587500e+01, 1.518750e+01, 3.950000e+01], [1.018750e+01, -1.737500e+01, 8.562500e+00, 3.406250e+00, -9.937500e+00, -1.156250e+01], [7.906250e+00, -8.187500e+00, 1.081250e+01, 5.250000e+00, -2.468750e+00, -6.562500e+00], [-2.312500e+00, 1.412500e+01, 1.006250e+01, 1.056250e+01, 1.568750e+01, -1.462500e+01]]> : tensor<4x6xbf16>
    return %cst : tensor<4x6xbf16>
  }
}
