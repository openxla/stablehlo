// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x1x6xbf16>, tensor<2x4x6xbf16>)
    %1 = call @expected() : () -> tensor<2x4x6xbf16>
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2 = stablehlo.pad %0#1, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xbf16>, tensor<bf16>) -> tensor<2x4x6xbf16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %cst_0) <{window_dimensions = array<i64: 1, 3, 1>, window_strides = array<i64: 1, 2, 1>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) : (tensor<2x4x6xbf16>, tensor<2x1x6xbf16>, tensor<bf16>) -> tensor<2x4x6xbf16>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xbf16>) -> tensor<2x4x6xbf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x4x6xbf16>, tensor<2x4x6xbf16>) -> ()
    return %4 : tensor<2x4x6xbf16>
  }
  func.func private @inputs() -> (tensor<2x1x6xbf16> {mhlo.layout_mode = "default"}, tensor<2x4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-4.937500e+00, -1.914060e+00, -1.093750e+00, -1.789060e+00, -5.281250e+00, 2.703130e+00]], [[5.859380e-01, -3.125000e+00, -1.640630e+00, 2.046880e+00, -1.117190e+00, -1.351560e+00]]]> : tensor<2x1x6xbf16>
    %cst_0 = stablehlo.constant dense<[[[-6.500000e+00, 5.500000e+00, -3.491210e-02, 2.421880e+00, 7.148430e-01, 1.890630e+00], [-4.687500e+00, -5.281250e+00, 1.726560e+00, -9.335930e-01, -1.070310e+00, -5.000000e-01], [8.789060e-01, -1.515630e+00, -3.457030e-01, 1.320310e+00, -1.242190e+00, -2.625000e+00], [3.156250e+00, 1.781250e+00, -1.664060e+00, 5.531250e+00, -1.242190e+00, 7.625000e+00]], [[-2.375000e+00, -1.960940e+00, -1.679690e+00, -1.921880e+00, -3.437500e-01, 1.906250e+00], [1.890630e+00, -1.859380e+00, 1.031250e+00, -1.953130e-01, -1.904300e-01, -8.593750e-01], [-3.750000e+00, -1.826170e-01, 5.625000e+00, -1.078130e+00, -2.843750e+00, 9.125000e+00], [-3.466800e-02, -5.531250e+00, -4.687500e+00, -3.859380e+00, 1.546880e+00, -2.765630e+00]]]> : tensor<2x4x6xbf16>
    return %cst, %cst_0 : tensor<2x1x6xbf16>, tensor<2x4x6xbf16>
  }
  func.func private @expected() -> (tensor<2x4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.000000e+00, -1.914060e+00, 0.000000e+00, -1.789060e+00, -5.281250e+00, 2.703130e+00], [0.000000e+00, 0.000000e+00, -1.093750e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-4.937500e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [5.859380e-01, 0.000000e+00, 0.000000e+00, 2.046880e+00, -1.117190e+00, 0.000000e+00], [0.000000e+00, -3.125000e+00, -1.640630e+00, 0.000000e+00, 0.000000e+00, -1.351560e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xbf16>
    return %cst : tensor<2x4x6xbf16>
  }
}
