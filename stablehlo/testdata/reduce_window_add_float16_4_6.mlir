// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xf16>
    %1 = call @expected() : () -> tensor<3x5xf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<f16>
    %3 = "stablehlo.reduce_window"(%0, %2) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %4 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %4 : tensor<f16>
    }) : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<3x5xf16>, tensor<3x5xf16>) -> ()
    return %3 : tensor<3x5xf16>
  }
  func.func private @inputs() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.099610e+00, -5.273440e+00, -1.480470e+00, 1.727540e+00, -2.679690e+00, 1.786130e+00], [1.854490e+00, 3.128910e+00, -8.891600e-01, 1.997070e+00, -6.285150e+00, -3.148440e+00], [7.609380e+00, 1.309570e+00, -1.653320e+00, -2.687500e+00, 1.111330e+00, 4.589840e+00], [-3.228520e+00, -2.228520e+00, 5.283200e-01, -3.468750e+00, -4.370120e-01, 1.621090e+00]]> : tensor<4x6xf16>
    return %cst : tensor<4x6xf16>
  }
  func.func private @expected() -> (tensor<3x5xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.390630e+00, -4.515630e+00, 1.355470e+00, -5.242190e+00, -1.032810e+01], [1.390630e+01, 1.897460e+00, -3.234380e+00, -5.867190e+00, -3.738280e+00], [3.466800e+00, -2.042970e+00, -7.281250e+00, -5.484380e+00, 6.886710e+00]]> : tensor<3x5xf16>
    return %cst : tensor<3x5xf16>
  }
}
