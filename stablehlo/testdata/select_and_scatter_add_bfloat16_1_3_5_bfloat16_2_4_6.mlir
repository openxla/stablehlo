// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x3x5xbf16>, tensor<2x4x6xbf16>)
    %1 = call @expected() : () -> tensor<2x4x6xbf16>
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2 = stablehlo.pad %0#1, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xbf16>, tensor<bf16>) -> tensor<2x4x6xbf16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %cst_0) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) : (tensor<2x4x6xbf16>, tensor<1x3x5xbf16>, tensor<bf16>) -> tensor<2x4x6xbf16>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xbf16>) -> tensor<2x4x6xbf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x4x6xbf16>, tensor<2x4x6xbf16>) -> ()
    return %4 : tensor<2x4x6xbf16>
  }
  func.func private @inputs() -> (tensor<1x3x5xbf16> {mhlo.layout_mode = "default"}, tensor<2x4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-2.140630e+00, 1.750000e+00, -6.562500e-01, -1.460940e+00, 3.437500e+00], [2.359380e+00, 1.835940e+00, 4.941410e-01, -1.085940e+00, -7.750000e+00], [-4.218750e+00, 1.968750e+00, -2.171880e+00, -1.945310e+00, 2.890630e-01]]]> : tensor<1x3x5xbf16>
    %cst_0 = stablehlo.constant dense<[[[1.591800e-01, -4.218750e+00, 2.484380e+00, -5.093750e+00, 3.203130e+00, 2.250000e+00], [1.226560e+00, -2.617190e-01, -1.664060e+00, -3.222660e-01, -1.554690e+00, -1.507810e+00], [3.968750e+00, -6.562500e-01, 6.523440e-01, 1.445310e-01, 1.937500e+00, -3.484380e+00], [-5.093750e+00, 6.406250e-01, -8.281250e-01, 3.328130e+00, 4.812500e+00, 3.062500e+00]], [[4.277340e-01, 8.359380e-01, -2.656250e+00, 1.726560e+00, 1.054690e+00, -1.914060e+00], [6.523440e-01, 3.656250e+00, -1.445310e+00, -1.164060e+00, -7.375000e+00, 3.656250e+00], [7.156250e+00, 2.546880e+00, 3.593750e+00, 4.343750e+00, -1.554690e+00, -8.945310e-01], [3.031250e+00, -2.125000e+00, -8.437500e-01, 3.093750e+00, -8.046880e-01, 1.226560e+00]]]> : tensor<2x4x6xbf16>
    return %cst, %cst_0 : tensor<1x3x5xbf16>, tensor<2x4x6xbf16>
  }
  func.func private @expected() -> (tensor<2x4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, -6.562500e-01, 0.000000e+00, -1.460940e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -1.656250e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.445310e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -4.312500e+00], [-1.859380e+00, 0.000000e+00, 1.968750e+00, -2.765630e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xbf16>
    return %cst : tensor<2x4x6xbf16>
  }
}
