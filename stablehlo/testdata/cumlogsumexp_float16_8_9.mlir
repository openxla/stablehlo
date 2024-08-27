// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xf16>
    %1 = call @expected() : () -> tensor<8x9xf16>
    %2 = call @cumlogsumexp(%0) : (tensor<8x9xf16>) -> tensor<8x9xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<8x9xf16>, tensor<8x9xf16>) -> ()
    return %2 : tensor<8x9xf16>
  }
  func.func private @inputs() -> (tensor<8x9xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.640630e+00, 6.402340e+00, -4.028320e-02, -1.413090e+00, -5.629880e-01, 2.050780e+00, 3.708980e+00, 2.750000e+00, 1.079100e+00], [3.837890e+00, -3.658200e+00, -3.484380e+00, -2.236330e-01, -1.178710e+00, 2.437500e+00, -3.763670e+00, 3.510740e-01, 1.617190e+00], [9.492180e-01, -4.882810e+00, -3.216800e+00, -4.730470e+00, -1.440430e+00, 1.725590e+00, -2.914060e+00, -4.480470e+00, 1.105470e+00], [-2.306640e+00, -1.630860e+00, -4.378910e+00, -4.628910e+00, -1.329100e+00, 7.280270e-01, 3.082030e+00, 5.371090e-01, 2.853520e+00], [-4.117190e+00, -4.509280e-01, 1.445310e-01, -6.613280e+00, 6.882810e+00, -1.913090e+00, 9.902340e-01, -4.788210e-02, -1.707760e-01], [1.004880e+00, -2.925780e+00, -5.296880e+00, -1.893620e-02, 4.046880e+00, -1.397460e+00, 2.890630e+00, 1.729490e+00, -1.813480e+00], [3.474610e+00, 1.231450e+00, -1.345700e+00, -1.536130e+00, 3.483890e-01, -3.669430e-01, -4.296880e+00, 9.360350e-01, 2.642580e+00], [-1.554690e+00, 1.316410e+00, -2.859380e+00, 6.425780e+00, -1.212890e+00, 4.617190e+00, 3.278810e-01, -3.203130e+00, -1.007810e+00]]> : tensor<8x9xf16>
    return %cst : tensor<8x9xf16>
  }
  func.func private @expected() -> (tensor<8x9xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.640630e+00, 6.402340e+00, -4.028320e-02, -1.413090e+00, -5.629880e-01, 2.050780e+00, 3.708980e+00, 2.750000e+00, 1.079100e+00], [3.839840e+00, 6.402340e+00, -8.819580e-03, 4.223630e-02, -1.311040e-01, 2.957030e+00, 3.708980e+00, 2.837890e+00, 2.078130e+00], [3.894530e+00, 6.402340e+00, 3.085330e-02, 5.065920e-02, 1.079100e-01, 3.212890e+00, 3.710940e+00, 2.837890e+00, 2.398440e+00], [3.896480e+00, 6.402340e+00, 4.293820e-02, 5.990600e-02, 3.210450e-01, 3.292970e+00, 4.136720e+00, 2.933590e+00, 3.345700e+00], [3.896480e+00, 6.402340e+00, 7.880860e-01, 6.115720e-02, 6.882810e+00, 3.298830e+00, 4.179690e+00, 2.982420e+00, 3.375000e+00], [3.951170e+00, 6.402340e+00, 7.905270e-01, 7.148430e-01, 6.941400e+00, 3.308590e+00, 4.421880e+00, 3.234380e+00, 3.380860e+00], [4.433590e+00, 6.406250e+00, 9.023430e-01, 8.149410e-01, 6.941400e+00, 3.333980e+00, 4.421880e+00, 3.330080e+00, 3.771480e+00], [4.437500e+00, 6.414060e+00, 9.252920e-01, 6.429690e+00, 6.941400e+00, 4.863280e+00, 4.437500e+00, 3.332030e+00, 3.779300e+00]]> : tensor<8x9xf16>
    return %cst : tensor<8x9xf16>
  }
  func.func private @cumlogsumexp(%arg0: tensor<8x9xf16>) -> tensor<8x9xf16> {
    %cst = stablehlo.constant dense<0xFC00> : tensor<f16>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
      %1 = func.call @logaddexp(%arg1, %arg2) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      stablehlo.return %1 : tensor<f16>
    }) : (tensor<8x9xf16>, tensor<f16>) -> tensor<8x9xf16>
    return %0 : tensor<8x9xf16>
  }
  func.func private @logaddexp(%arg0: tensor<f16> {mhlo.layout_mode = "default"}, %arg1: tensor<f16> {mhlo.layout_mode = "default"}) -> (tensor<f16> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<f16>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<f16>
    %4 = stablehlo.abs %1 : tensor<f16>
    %5 = stablehlo.negate %4 : tensor<f16>
    %6 = stablehlo.exponential %5 : tensor<f16>
    %7 = stablehlo.log_plus_one %6 : tensor<f16>
    %8 = stablehlo.add %0, %7 : tensor<f16>
    %9 = stablehlo.select %2, %3, %8 : tensor<i1>, tensor<f16>
    return %9 : tensor<f16>
  }
}
