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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xf16>, tensor<2x1xi32>, tensor<3x2x4xf16>) -> tensor<3x5x4xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xf16>, tensor<3x5x4xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xf16>, tensor<3x2x4xf16>) {
    %0 = stablehlo.constant dense<[[[-2.234380e+00, -3.529300e+00, -3.152340e+00, -4.769530e+00], [-1.676760e+00, 7.082030e+00, 2.400390e+00, 3.082030e+00], [-2.152340e+00, -2.835940e+00, -2.494140e+00, 3.322270e+00], [3.421880e+00, -2.511720e+00, 3.129880e-01, -3.212890e+00], [3.253170e-02, -1.664060e+00, -1.237300e+00, 3.244140e+00]], [[1.688480e+00, 4.394530e+00, 2.625000e+00, 3.505860e-01], [-6.318360e-01, 3.341800e+00, 3.304690e+00, -2.943360e+00], [2.816410e+00, -1.152340e+00, -5.464840e+00, -8.120110e-01], [-4.070310e+00, -2.507810e+00, -9.526360e-01, -3.623050e-01], [-1.872070e+00, 3.623050e-01, -3.623050e+00, -2.376950e+00]], [[-3.175780e+00, -1.883790e+00, 2.517580e+00, -1.203130e+00], [-8.552550e-03, -1.066410e+00, 5.029300e-01, -3.740230e+00], [-3.134770e+00, 1.586910e+00, -1.135740e+00, -2.856450e-01], [1.483150e-01, 5.035160e+00, 9.980460e-01, 6.066400e+00], [1.486330e+00, -4.031250e+00, -2.115480e-01, 7.421880e-01]]]> : tensor<3x5x4xf16>
    %1 = stablehlo.constant dense<[[[-1.433590e+00, -2.012630e-02, 8.139640e-01, 2.906250e+00], [5.839840e+00, 1.997070e+00, 3.423830e+00, -4.941410e+00]], [[2.050780e+00, 4.949220e+00, -7.429680e+00, -1.035160e+00], [9.384760e-01, 7.807610e-01, -3.302730e+00, 2.861330e+00]], [[9.829100e-01, -8.540030e-01, 9.082030e-01, -1.989750e-02], [-7.226560e-01, 3.041990e-01, 9.521480e-01, -1.445310e+00]]]> : tensor<3x2x4xf16>
    return %0, %1 : tensor<3x5x4xf16>, tensor<3x2x4xf16>
  }
  func.func private @expected() -> tensor<3x5x4xf16> {
    %0 = stablehlo.constant dense<[[[-2.234380e+00, -3.529300e+00, -3.152340e+00, -4.769530e+00], [5.839840e+00, 7.082030e+00, 3.423830e+00, 3.082030e+00], [-2.152340e+00, -2.835940e+00, -2.494140e+00, 3.322270e+00], [3.421880e+00, -2.511720e+00, 3.129880e-01, -3.212890e+00], [3.253170e-02, -1.664060e+00, -1.237300e+00, 3.244140e+00]], [[1.688480e+00, 4.394530e+00, 2.625000e+00, 3.505860e-01], [2.050780e+00, 4.949220e+00, 3.304690e+00, 2.861330e+00], [2.816410e+00, -1.152340e+00, -5.464840e+00, -8.120110e-01], [-4.070310e+00, -2.507810e+00, -9.526360e-01, -3.623050e-01], [-1.872070e+00, 3.623050e-01, -3.623050e+00, -2.376950e+00]], [[-3.175780e+00, -1.883790e+00, 2.517580e+00, -1.203130e+00], [9.829100e-01, 3.041990e-01, 9.521480e-01, -1.989750e-02], [-3.134770e+00, 1.586910e+00, -1.135740e+00, -2.856450e-01], [1.483150e-01, 5.035160e+00, 9.980460e-01, 6.066400e+00], [1.486330e+00, -4.031250e+00, -2.115480e-01, 7.421880e-01]]]> : tensor<3x5x4xf16>
    return %0 : tensor<3x5x4xf16>
  }
}

