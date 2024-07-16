// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi16>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi16>) -> tensor<4x3xf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    return %4 : tensor<4x6xf16>
  }
  func.func private @inputs() -> (tensor<4x3xi16> {mhlo.layout_mode = "default"}, tensor<3x6xf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 0, 2], [7, -2, -1], [-5, 0, 0], [3, -2, 4]]> : tensor<4x3xi16>
    %cst = stablehlo.constant dense<[[-4.972660e+00, 9.633780e-01, 9.179680e-01, 2.687500e+00, 1.023440e+00, -1.614260e+00], [7.612300e-01, 3.560550e+00, -1.004880e+00, 1.511720e+00, 3.806640e+00, -6.101560e+00], [-5.164060e+00, 5.737300e-01, -3.597660e+00, 4.582030e+00, -2.884770e+00, 1.960940e+00]]> : tensor<3x6xf16>
    return %c, %cst : tensor<4x3xi16>, tensor<3x6xf16>
  }
  func.func private @expected() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.032810e+01, 1.147460e+00, -7.195310e+00, 9.164060e+00, -5.769530e+00, 3.921880e+00], [-3.117190e+01, -9.511710e-01, 1.203130e+01, 1.120310e+01, 2.435550e+00, -1.057620e+00], [2.485940e+01, -4.816410e+00, -4.589840e+00, -1.343750e+01, -5.117190e+00, 8.070310e+00], [-3.709380e+01, -1.935550e+00, -9.625000e+00, 2.337500e+01, -1.607810e+01, 1.520310e+01]]> : tensor<4x6xf16>
    return %cst : tensor<4x6xf16>
  }
}
