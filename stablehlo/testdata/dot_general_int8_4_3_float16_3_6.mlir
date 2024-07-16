// RUN-DISABLED(inaccurate) stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi8>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi8>) -> tensor<4x3xf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    stablehlo.custom_call @check.expect_almost_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    return %4 : tensor<4x6xf16>
  }
  func.func private @inputs() -> (tensor<4x3xi8> {mhlo.layout_mode = "default"}, tensor<3x6xf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-1, 1, -6], [2, -3, 5], [-2, -4, -3], [-1, 0, 0]]> : tensor<4x3xi8>
    %cst = stablehlo.constant dense<[[-3.587890e+00, -4.755860e-01, -3.792970e+00, 3.625000e+00, -3.273930e-01, -6.378900e+00], [3.304690e+00, -2.542970e+00, -3.156250e+00, -5.953130e+00, -3.101560e+00, 1.036130e+00], [-6.839840e+00, -2.187500e+00, -4.273440e+00, -4.761720e+00, -9.516600e-01, 7.207030e-01]]> : tensor<3x6xf16>
    return %c, %cst : tensor<4x3xi8>, tensor<3x6xf16>
  }
  func.func private @expected() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.793750e+01, 1.105470e+01, 2.628130e+01, 1.900000e+01, 2.935550e+00, 3.089840e+00], [-5.128130e+01, -4.257810e+00, -1.948440e+01, 1.300780e+00, 3.890630e+00, -1.226560e+01], [1.447660e+01, 1.768750e+01, 3.303130e+01, 3.084380e+01, 1.591410e+01, 6.453130e+00], [3.587890e+00, 4.755860e-01, 3.792970e+00, -3.625000e+00, 3.273930e-01, 6.378900e+00]]> : tensor<4x6xf16>
    return %cst : tensor<4x6xf16>
  }
}
