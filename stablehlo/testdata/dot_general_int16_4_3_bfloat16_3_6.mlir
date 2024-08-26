// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi16>, tensor<3x6xbf16>)
    %1 = call @expected() : () -> tensor<4x6xbf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi16>) -> tensor<4x3xbf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xbf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    return %4 : tensor<4x6xbf16>
  }
  func.func private @inputs() -> (tensor<4x3xi16> {mhlo.layout_mode = "default"}, tensor<3x6xbf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, -1, -2], [-3, 3, 2], [0, -3, 0], [-2, 2, 0]]> : tensor<4x3xi16>
    %cst = stablehlo.constant dense<[[1.187500e+00, 5.906250e+00, 5.406250e+00, -5.898440e-01, 2.281250e+00, 1.552730e-01], [1.609380e+00, 1.273440e+00, -3.937500e+00, 6.562500e-01, 1.015630e+00, -1.777340e-01], [1.804690e+00, -2.406250e+00, -2.968750e-01, 3.015630e+00, -5.312500e+00, 2.828130e+00]]> : tensor<3x6xbf16>
    return %c, %cst : tensor<4x3xi16>, tensor<3x6xbf16>
  }
  func.func private @expected() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.843750e+00, 1.537500e+01, 1.537500e+01, -7.875000e+00, 1.418750e+01, -5.156250e+00], [4.875000e+00, -1.875000e+01, -2.862500e+01, 9.750000e+00, -1.443750e+01, 4.656250e+00], [-4.812500e+00, -3.812500e+00, 1.181250e+01, -1.968750e+00, -3.046880e+00, 5.312500e-01], [8.437500e-01, -9.250000e+00, -1.875000e+01, 2.500000e+00, -2.531250e+00, -6.640630e-01]]> : tensor<4x6xbf16>
    return %cst : tensor<4x6xbf16>
  }
}
