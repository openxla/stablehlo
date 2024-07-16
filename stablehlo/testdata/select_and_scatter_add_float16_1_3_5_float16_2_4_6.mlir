// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x3x5xf16>, tensor<2x4x6xf16>)
    %1 = call @expected() : () -> tensor<2x4x6xf16>
    %cst = stablehlo.constant dense<0xFC00> : tensor<f16>
    %2 = stablehlo.pad %0#1, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xf16>, tensor<f16>) -> tensor<2x4x6xf16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %cst_0) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) : (tensor<2x4x6xf16>, tensor<1x3x5xf16>, tensor<f16>) -> tensor<2x4x6xf16>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xf16>) -> tensor<2x4x6xf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x4x6xf16>, tensor<2x4x6xf16>) -> ()
    return %4 : tensor<2x4x6xf16>
  }
  func.func private @inputs() -> (tensor<1x3x5xf16> {mhlo.layout_mode = "default"}, tensor<2x4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-3.878910e+00, 1.413090e+00, 4.476560e+00, -3.933590e+00, 6.054690e-02], [-7.019530e+00, -1.442380e+00, 4.257810e+00, -3.005860e+00, 2.126950e+00], [4.484380e+00, -3.417970e-02, -1.207030e+00, 4.562500e+00, 3.677730e+00]]]> : tensor<1x3x5xf16>
    %cst_0 = stablehlo.constant dense<[[[-1.462890e+00, -3.843750e+00, 2.527340e+00, -5.609380e+00, -3.707030e+00, -2.681640e+00], [-1.667970e+00, 1.726560e+00, 3.255860e+00, 5.164060e+00, -1.097660e+00, 1.143190e-01], [-2.068360e+00, -1.214840e+00, -4.921880e+00, 1.913090e+00, -3.544920e+00, 3.684080e-01], [-1.171880e-01, -4.000000e+00, -1.446290e+00, -5.804690e+00, -6.160150e+00, -4.410160e+00]], [[-2.937500e+00, 1.557920e-02, 7.915030e-01, 5.996090e+00, 4.687500e+00, -3.720700e+00], [3.645020e-01, -6.455080e-01, 3.906250e+00, 5.367190e+00, 2.393800e-01, 1.419680e-01], [1.138670e+00, 2.181640e+00, -2.730470e+00, 5.585940e-01, 1.885740e+00, -3.236330e+00], [-4.449220e+00, -1.608400e+00, 9.716790e-01, -6.566400e+00, 2.250000e+00, 3.755860e+00]]]> : tensor<2x4x6xf16>
    return %cst, %cst_0 : tensor<1x3x5xf16>, tensor<2x4x6xf16>
  }
  func.func private @expected() -> (tensor<2x4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, -3.878910e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, -1.207030e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 5.429690e-01, 6.054690e-02, 0.000000e+00], [0.000000e+00, 0.000000e+00, -2.929690e-02, 1.251950e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, -2.570310e+00, 0.000000e+00, 0.000000e+00, 2.126950e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 4.562500e+00, 3.677730e+00]]]> : tensor<2x4x6xf16>
    return %cst : tensor<2x4x6xf16>
  }
}
