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
    %0 = stablehlo.constant dense<[[[-2.224610e+00, 5.570310e+00, 2.273440e+00, -2.404300e+00], [-2.291020e+00, -4.536130e-01, -9.506830e-01, 3.894530e+00], [4.574220e+00, -3.427120e-02, -2.835940e+00, -2.825930e-02], [2.703130e+00, -1.018550e+00, 3.130860e+00, 3.242190e+00], [-5.734380e+00, 5.085940e+00, 3.933590e+00, -2.011720e+00]], [[2.880860e+00, 1.456050e+00, -2.703130e+00, -1.502930e+00], [-3.277340e+00, -2.982420e+00, 1.546880e+00, -1.331790e-01], [3.227540e-01, 2.314450e+00, 6.503900e+00, 1.446290e+00], [-2.324220e+00, 1.653320e+00, 5.312500e+00, 1.214840e+00], [3.684080e-01, -1.430660e+00, -7.670890e-01, 7.592770e-01]], [[1.284180e+00, 3.763670e+00, 1.668950e+00, 2.159420e-01], [-1.928710e+00, -3.652340e+00, -3.884770e+00, 1.104490e+00], [1.846680e+00, 1.280270e+00, -1.908200e+00, -4.035640e-01], [4.609380e+00, -1.169920e+00, -5.717770e-01, -4.656250e+00], [4.946290e-01, 1.553710e+00, -2.246090e+00, -1.727540e+00]]]> : tensor<3x5x4xf16>
    %1 = stablehlo.constant dense<[[[-2.435550e+00, 1.169920e+00, -3.878910e+00, 3.970700e+00], [1.498050e+00, -1.068360e+00, 3.046880e+00, -4.230470e+00]], [[-4.335940e+00, 1.121090e+00, 4.171880e+00, -3.276370e-01], [-1.181640e+00, 1.652340e+00, 1.208010e+00, -3.843750e+00]], [[2.833980e+00, -3.709790e-03, 1.850590e+00, 2.998050e+00], [6.273440e+00, -3.898440e+00, 5.425780e+00, -3.216800e+00]]]> : tensor<3x2x4xf16>
    return %0, %1 : tensor<3x5x4xf16>, tensor<3x2x4xf16>
  }
  func.func private @expected() -> tensor<3x5x4xf16> {
    %0 = stablehlo.constant dense<[[[-2.224610e+00, 5.570310e+00, 2.273440e+00, -2.404300e+00], [-3.228520e+00, -3.520510e-01, -1.781250e+00, 3.636720e+00], [4.574220e+00, -3.427120e-02, -2.835940e+00, -2.825930e-02], [2.703130e+00, -1.018550e+00, 3.130860e+00, 3.242190e+00], [-5.734380e+00, 5.085940e+00, 3.933590e+00, -2.011720e+00]], [[2.880860e+00, 1.456050e+00, -2.703130e+00, -1.502930e+00], [-8.796870e+00, -2.089840e-01, 6.925780e+00, -4.304690e+00], [3.227540e-01, 2.314450e+00, 6.503900e+00, 1.446290e+00], [-2.324220e+00, 1.653320e+00, 5.312500e+00, 1.214840e+00], [3.684080e-01, -1.430660e+00, -7.670890e-01, 7.592770e-01]], [[1.284180e+00, 3.763670e+00, 1.668950e+00, 2.159420e-01], [7.179680e+00, -7.554680e+00, 3.390630e+00, 8.847650e-01], [1.846680e+00, 1.280270e+00, -1.908200e+00, -4.035640e-01], [4.609380e+00, -1.169920e+00, -5.717770e-01, -4.656250e+00], [4.946290e-01, 1.553710e+00, -2.246090e+00, -1.727540e+00]]]> : tensor<3x5x4xf16>
    return %0 : tensor<3x5x4xf16>
  }
}

