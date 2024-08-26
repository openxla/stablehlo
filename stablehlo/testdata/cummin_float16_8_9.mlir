// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xf16>
    %1 = call @expected() : () -> tensor<8x9xf16>
    %2 = call @cummin(%0) : (tensor<8x9xf16>) -> tensor<8x9xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<8x9xf16>, tensor<8x9xf16>) -> ()
    return %2 : tensor<8x9xf16>
  }
  func.func private @inputs() -> (tensor<8x9xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.523930e-01, -2.701170e+00, -3.048830e+00, -1.825200e+00, -1.894530e+00, 1.485600e-01, 3.723140e-01, -1.048830e+00, 3.082030e+00], [2.759770e+00, -1.254880e-01, 4.125000e+00, 7.817380e-01, 6.928710e-01, 2.568360e+00, 1.751950e+00, -5.917960e+00, -1.809570e+00], [1.776370e+00, 6.613280e+00, 2.546880e+00, 2.539060e+00, -3.582030e+00, -4.117190e+00, -4.593750e+00, 2.309570e-01, -6.108400e-01], [5.810550e-01, 2.279300e+00, -2.294920e+00, -2.169920e+00, 3.250000e+00, -2.380860e+00, 4.480470e+00, 1.847660e+00, 2.556640e+00], [-4.417970e+00, -1.994140e+00, 9.731440e-01, -4.027340e+00, -1.264650e+00, -5.699210e+00, 1.718750e+00, 3.837890e+00, -3.726560e+00], [1.138670e+00, 4.597660e+00, 2.812500e+00, 2.634280e-01, 2.766110e-01, -1.526370e+00, 1.711910e+00, -5.507810e+00, -1.961910e+00], [7.402340e+00, 3.042970e+00, -5.925780e+00, 1.907230e+00, 3.675780e+00, 6.582030e-01, -9.565420e-01, -1.853520e+00, 5.605460e+00], [-1.355470e+00, -1.752930e+00, -4.484380e+00, -6.166990e-01, 3.089840e+00, 4.488280e+00, 1.602540e+00, -1.615230e+00, -1.755860e+00]]> : tensor<8x9xf16>
    return %cst : tensor<8x9xf16>
  }
  func.func private @expected() -> (tensor<8x9xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.523930e-01, -2.701170e+00, -3.048830e+00, -1.825200e+00, -1.894530e+00, 1.485600e-01, 3.723140e-01, -1.048830e+00, 3.082030e+00], [4.523930e-01, -2.701170e+00, -3.048830e+00, -1.825200e+00, -1.894530e+00, 1.485600e-01, 3.723140e-01, -5.917960e+00, -1.809570e+00], [4.523930e-01, -2.701170e+00, -3.048830e+00, -1.825200e+00, -3.582030e+00, -4.117190e+00, -4.593750e+00, -5.917960e+00, -1.809570e+00], [4.523930e-01, -2.701170e+00, -3.048830e+00, -2.169920e+00, -3.582030e+00, -4.117190e+00, -4.593750e+00, -5.917960e+00, -1.809570e+00], [-4.417970e+00, -2.701170e+00, -3.048830e+00, -4.027340e+00, -3.582030e+00, -5.699210e+00, -4.593750e+00, -5.917960e+00, -3.726560e+00], [-4.417970e+00, -2.701170e+00, -3.048830e+00, -4.027340e+00, -3.582030e+00, -5.699210e+00, -4.593750e+00, -5.917960e+00, -3.726560e+00], [-4.417970e+00, -2.701170e+00, -5.925780e+00, -4.027340e+00, -3.582030e+00, -5.699210e+00, -4.593750e+00, -5.917960e+00, -3.726560e+00], [-4.417970e+00, -2.701170e+00, -5.925780e+00, -4.027340e+00, -3.582030e+00, -5.699210e+00, -4.593750e+00, -5.917960e+00, -3.726560e+00]]> : tensor<8x9xf16>
    return %cst : tensor<8x9xf16>
  }
  func.func private @cummin(%arg0: tensor<8x9xf16>) -> tensor<8x9xf16> {
    %cst = stablehlo.constant dense<0x7C00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<f16>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
      %2 = stablehlo.minimum %arg1, %arg2 : tensor<f16>
      stablehlo.return %2 : tensor<f16>
    }) : (tensor<8x9xf16>, tensor<f16>) -> tensor<8x9xf16>
    return %1 : tensor<8x9xf16>
  }
}
