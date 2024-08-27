// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui64>, tensor<3x6xbf16>)
    %1 = call @expected() : () -> tensor<4x6xbf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui64>) -> tensor<4x3xbf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xbf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    return %4 : tensor<4x6xbf16>
  }
  func.func private @inputs() -> (tensor<4x3xui64> {mhlo.layout_mode = "default"}, tensor<3x6xbf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[5, 1, 3], [5, 0, 5], [3, 1, 2], [1, 3, 2]]> : tensor<4x3xui64>
    %cst = stablehlo.constant dense<[[1.429690e+00, -1.265630e+00, -1.318360e-01, -1.101560e+00, 2.796880e+00, 2.421880e+00], [-3.339840e-01, 2.734380e+00, 4.062500e+00, -1.867190e+00, -1.117190e+00, -1.281250e+00], [2.250000e+00, -3.125000e+00, 1.476560e+00, -8.007810e-02, 3.765630e+00, 9.609370e-01]]> : tensor<3x6xbf16>
    return %c, %cst : tensor<4x3xui64>, tensor<3x6xbf16>
  }
  func.func private @expected() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.356250e+01, -1.300000e+01, 7.843750e+00, -7.625000e+00, 2.412500e+01, 1.368750e+01], [1.837500e+01, -2.200000e+01, 6.718750e+00, -5.906250e+00, 3.275000e+01, 1.687500e+01], [8.437500e+00, -7.312500e+00, 6.625000e+00, -5.343750e+00, 1.481250e+01, 7.906250e+00], [4.937500e+00, 6.875000e-01, 1.500000e+01, -6.875000e+00, 6.968750e+00, 5.000000e-01]]> : tensor<4x6xbf16>
    return %cst : tensor<4x6xbf16>
  }
}
