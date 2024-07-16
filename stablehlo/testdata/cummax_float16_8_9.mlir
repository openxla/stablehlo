// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xf16>
    %1 = call @expected() : () -> tensor<8x9xf16>
    %2 = call @cummax(%0) : (tensor<8x9xf16>) -> tensor<8x9xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<8x9xf16>, tensor<8x9xf16>) -> ()
    return %2 : tensor<8x9xf16>
  }
  func.func private @inputs() -> (tensor<8x9xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-6.186520e-01, 3.033200e+00, 2.978520e+00, -4.535160e+00, -3.171880e+00, -5.035160e+00, -2.421880e+00, -3.779300e+00, -2.730470e+00], [3.623050e-01, 7.143550e-01, -4.710940e+00, -4.199220e-01, 2.601560e+00, -9.887690e-01, 8.403320e-01, 1.086430e-01, 2.517090e-01], [-3.406250e+00, -3.289060e+00, -1.094730e+00, -5.014650e-01, -2.966800e+00, 5.229490e-01, 1.628910e+00, 1.155270e+00, -2.535160e+00], [-2.736330e+00, -6.425780e-01, -9.780270e-01, 4.855470e+00, -1.765630e+00, -1.186520e+00, 8.728020e-02, -2.748050e+00, -5.414060e+00], [-1.822270e+00, 2.070310e+00, -6.689450e-02, -4.614260e-01, 2.667970e+00, 1.663090e+00, 3.801270e-01, 1.887700e+00, 4.175780e+00], [1.050780e+00, -7.335930e+00, -1.176760e+00, 3.509770e+00, -2.902340e+00, 1.328130e+00, -4.062500e+00, -1.000980e+00, -1.870120e+00], [-9.619140e-01, 9.355460e-01, -5.166020e-01, 2.509770e+00, 5.041500e-02, 3.617190e+00, 1.008790e+00, 2.894530e+00, -2.992190e+00], [1.180660e+00, -1.629880e+00, 1.940430e+00, -3.410160e+00, -7.640630e+00, 6.352540e-01, -2.593990e-02, -5.308590e+00, 2.566410e+00]]> : tensor<8x9xf16>
    return %cst : tensor<8x9xf16>
  }
  func.func private @expected() -> (tensor<8x9xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-6.186520e-01, 3.033200e+00, 2.978520e+00, -4.535160e+00, -3.171880e+00, -5.035160e+00, -2.421880e+00, -3.779300e+00, -2.730470e+00], [3.623050e-01, 3.033200e+00, 2.978520e+00, -4.199220e-01, 2.601560e+00, -9.887690e-01, 8.403320e-01, 1.086430e-01, 2.517090e-01], [3.623050e-01, 3.033200e+00, 2.978520e+00, -4.199220e-01, 2.601560e+00, 5.229490e-01, 1.628910e+00, 1.155270e+00, 2.517090e-01], [3.623050e-01, 3.033200e+00, 2.978520e+00, 4.855470e+00, 2.601560e+00, 5.229490e-01, 1.628910e+00, 1.155270e+00, 2.517090e-01], [3.623050e-01, 3.033200e+00, 2.978520e+00, 4.855470e+00, 2.667970e+00, 1.663090e+00, 1.628910e+00, 1.887700e+00, 4.175780e+00], [1.050780e+00, 3.033200e+00, 2.978520e+00, 4.855470e+00, 2.667970e+00, 1.663090e+00, 1.628910e+00, 1.887700e+00, 4.175780e+00], [1.050780e+00, 3.033200e+00, 2.978520e+00, 4.855470e+00, 2.667970e+00, 3.617190e+00, 1.628910e+00, 2.894530e+00, 4.175780e+00], [1.180660e+00, 3.033200e+00, 2.978520e+00, 4.855470e+00, 2.667970e+00, 3.617190e+00, 1.628910e+00, 2.894530e+00, 4.175780e+00]]> : tensor<8x9xf16>
    return %cst : tensor<8x9xf16>
  }
  func.func private @cummax(%arg0: tensor<8x9xf16>) -> tensor<8x9xf16> {
    %cst = stablehlo.constant dense<0xFC00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<f16>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
      %2 = stablehlo.maximum %arg1, %arg2 : tensor<f16>
      stablehlo.return %2 : tensor<f16>
    }) : (tensor<8x9xf16>, tensor<f16>) -> tensor<8x9xf16>
    return %1 : tensor<8x9xf16>
  }
}
