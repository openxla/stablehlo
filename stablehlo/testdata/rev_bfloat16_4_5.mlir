// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x5xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x5xbf16>
    %1 = call @expected() : () -> tensor<4x5xbf16>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x5xbf16>, tensor<4x5xbf16>) -> ()
    return %2 : tensor<4x5xbf16>
  }
  func.func private @inputs() -> (tensor<4x5xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.937500e+00, 5.234380e-01, 4.625000e+00, 2.687500e+00, -3.937500e+00], [-3.734380e+00, 5.375000e+00, -5.437500e+00, 4.468750e+00, 3.250000e+00], [-1.656250e+00, 2.593750e+00, -4.531250e+00, 8.906250e-01, 6.601560e-01], [-3.156250e+00, -2.171880e+00, -3.328130e+00, 2.484380e+00, 7.187500e-01]]> : tensor<4x5xbf16>
    return %cst : tensor<4x5xbf16>
  }
  func.func private @expected() -> (tensor<4x5xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.156250e+00, -2.171880e+00, -3.328130e+00, 2.484380e+00, 7.187500e-01], [-1.656250e+00, 2.593750e+00, -4.531250e+00, 8.906250e-01, 6.601560e-01], [-3.734380e+00, 5.375000e+00, -5.437500e+00, 4.468750e+00, 3.250000e+00], [3.937500e+00, 5.234380e-01, 4.625000e+00, 2.687500e+00, -3.937500e+00]]> : tensor<4x5xbf16>
    return %cst : tensor<4x5xbf16>
  }
}
