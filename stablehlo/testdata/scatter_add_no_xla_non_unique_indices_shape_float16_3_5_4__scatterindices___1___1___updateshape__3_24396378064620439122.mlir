// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x4xf16>, tensor<3x2x4xf16>)
    %2 = call @expected() : () -> tensor<3x5x4xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xf16>, tensor<2x1xi32>, tensor<3x2x4xf16>) -> tensor<3x5x4xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xf16>, tensor<3x5x4xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xf16>, tensor<3x2x4xf16>) {
    %0 = stablehlo.constant dense<[[[2.826170e+00, 1.384770e+00, 3.515630e+00, -3.521480e+00], [-3.041990e-01, -4.402340e+00, -5.996090e+00, 1.034180e+00], [9.624020e-01, -5.480460e+00, 4.476560e+00, 4.807130e-01], [-2.830080e+00, -3.005370e-01, 3.515630e+00, 1.201170e+00], [-1.252930e+00, -3.732910e-01, -7.679680e+00, -4.117190e+00]], [[-1.187500e+00, 2.354740e-01, 1.047970e-01, -2.349610e+00], [1.958010e+00, 3.029790e-01, 1.704100e+00, -1.707760e-01], [-1.777340e+00, -2.992190e+00, -6.928710e-01, 3.406250e+00], [1.023440e+00, -3.507810e+00, -2.945310e+00, 4.187010e-01], [3.718260e-01, -4.887700e-01, -8.574210e-01, -6.527340e+00]], [[3.023440e+00, 1.372070e+00, -6.386710e+00, -2.810550e+00], [5.183590e+00, 2.705080e+00, -5.307620e-01, 2.988280e-01], [1.616210e+00, 1.341800e+00, -4.218750e-01, 1.302730e+00], [3.091800e+00, -3.753910e+00, 4.150390e-01, -1.576170e+00], [3.863280e+00, -5.253910e-01, 7.089840e-01, 3.539060e+00]]]> : tensor<3x5x4xf16>
    %1 = stablehlo.constant dense<[[[2.306640e+00, 1.174800e+00, -5.996090e-01, 2.205080e+00], [-3.068360e+00, 1.585940e+00, 2.179690e+00, 3.314450e+00]], [[-1.347660e+00, -4.339840e+00, -2.419920e+00, -5.131840e-01], [-1.008790e+00, 1.288090e+00, -1.942380e+00, -1.026370e+00]], [[-1.944340e+00, -6.953130e-01, -1.181640e+00, -1.612550e-01], [9.287100e-01, 4.318850e-01, -2.095700e+00, 6.402340e+00]]]> : tensor<3x2x4xf16>
    return %0, %1 : tensor<3x5x4xf16>, tensor<3x2x4xf16>
  }
  func.func private @expected() -> tensor<3x5x4xf16> {
    %0 = stablehlo.constant dense<[[[2.826170e+00, 1.384770e+00, 3.515630e+00, -3.521480e+00], [-1.066410e+00, -1.640630e+00, -4.414060e+00, 6.554690e+00], [9.624020e-01, -5.480460e+00, 4.476560e+00, 4.807130e-01], [-2.830080e+00, -3.005370e-01, 3.515630e+00, 1.201170e+00], [-1.252930e+00, -3.732910e-01, -7.679680e+00, -4.117190e+00]], [[-1.187500e+00, 2.354740e-01, 1.047970e-01, -2.349610e+00], [-3.984380e-01, -2.746090e+00, -2.658200e+00, -1.710940e+00], [-1.777340e+00, -2.992190e+00, -6.928710e-01, 3.406250e+00], [1.023440e+00, -3.507810e+00, -2.945310e+00, 4.187010e-01], [3.718260e-01, -4.887700e-01, -8.574210e-01, -6.527340e+00]], [[3.023440e+00, 1.372070e+00, -6.386710e+00, -2.810550e+00], [4.167970e+00, 2.441410e+00, -3.808590e+00, 6.539060e+00], [1.616210e+00, 1.341800e+00, -4.218750e-01, 1.302730e+00], [3.091800e+00, -3.753910e+00, 4.150390e-01, -1.576170e+00], [3.863280e+00, -5.253910e-01, 7.089840e-01, 3.539060e+00]]]> : tensor<3x5x4xf16>
    return %0 : tensor<3x5x4xf16>
  }
}

