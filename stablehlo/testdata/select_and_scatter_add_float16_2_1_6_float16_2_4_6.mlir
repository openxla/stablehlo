// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x1x6xf16>, tensor<2x4x6xf16>)
    %1 = call @expected() : () -> tensor<2x4x6xf16>
    %cst = stablehlo.constant dense<0xFC00> : tensor<f16>
    %2 = stablehlo.pad %0#1, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xf16>, tensor<f16>) -> tensor<2x4x6xf16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %cst_0) <{window_dimensions = array<i64: 1, 3, 1>, window_strides = array<i64: 1, 2, 1>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) : (tensor<2x4x6xf16>, tensor<2x1x6xf16>, tensor<f16>) -> tensor<2x4x6xf16>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xf16>) -> tensor<2x4x6xf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x4x6xf16>, tensor<2x4x6xf16>) -> ()
    return %4 : tensor<2x4x6xf16>
  }
  func.func private @inputs() -> (tensor<2x1x6xf16> {mhlo.layout_mode = "default"}, tensor<2x4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[4.429690e+00, -1.569340e+00, -1.449220e+00, 3.458980e+00, 1.494140e+00, 6.191400e-01]], [[-4.476560e+00, 5.312500e-01, -2.318120e-01, -2.810550e+00, -5.971680e-01, -3.281250e+00]]]> : tensor<2x1x6xf16>
    %cst_0 = stablehlo.constant dense<[[[1.683590e+00, 7.891850e-02, 4.265630e+00, -2.978520e+00, -4.772950e-02, -8.564450e-01], [3.792970e+00, -3.361330e+00, -9.091790e-01, 3.884770e+00, -6.233210e-03, 6.996090e+00], [-5.656250e+00, -3.251950e-01, 2.166020e+00, 3.173830e+00, -1.076170e+00, -3.181150e-01], [5.083010e-01, 2.471920e-01, -1.317380e+00, -5.496090e+00, 7.548830e-01, 3.498540e-01]], [[1.626950e+00, -1.175780e+00, -5.027340e+00, -6.410150e+00, -1.860350e+00, 4.449220e+00], [3.728520e+00, 3.277340e+00, 1.888670e+00, 1.446530e-01, -1.328130e+00, -6.481930e-02], [5.054690e+00, 5.511710e+00, -6.898430e+00, -4.750000e+00, -1.661130e+00, 2.384770e+00], [5.297850e-01, 5.062500e+00, -4.500000e+00, -2.199710e-01, -6.414060e+00, -2.808590e+00]]]> : tensor<2x4x6xf16>
    return %cst, %cst_0 : tensor<2x1x6xf16>, tensor<2x4x6xf16>
  }
  func.func private @expected() -> (tensor<2x4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.000000e+00, -1.569340e+00, -1.449220e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [4.429690e+00, 0.000000e+00, 0.000000e+00, 3.458980e+00, 1.494140e+00, 6.191400e-01], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -3.281250e+00], [0.000000e+00, 0.000000e+00, -2.318120e-01, -2.810550e+00, -5.971680e-01, 0.000000e+00], [-4.476560e+00, 5.312500e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf16>
    return %cst : tensor<2x4x6xf16>
  }
}
