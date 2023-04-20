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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xf16>, tensor<2x1xi32>, tensor<3x2x4xf16>) -> tensor<3x5x4xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xf16>, tensor<3x5x4xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xf16>, tensor<3x2x4xf16>) {
    %0 = stablehlo.constant dense<[[[-2.097660e+00, -1.900390e+00, -9.916990e-01, -6.335940e+00], [-3.164060e+00, 4.160160e+00, 3.393550e-01, 2.380860e+00], [-2.152340e+00, -4.033200e-01, -3.355470e+00, 1.889650e+00], [-4.093750e+00, 1.389650e+00, -3.257810e+00, 2.123050e+00], [3.828130e-01, 1.972660e+00, -1.303710e+00, 1.476560e+00]], [[3.105470e+00, -1.161130e+00, -5.024410e-01, -5.316410e+00], [-4.335940e+00, 2.857420e+00, 5.746090e+00, 1.020510e+00], [-4.214840e+00, -1.667970e+00, -2.269530e+00, -5.812500e+00], [4.072270e-01, -4.964840e+00, 3.513180e-01, -2.195310e+00], [5.844730e-01, -4.820310e+00, -8.486330e-01, 4.093750e+00]], [[-3.029790e-01, -4.013670e-01, -3.593750e+00, -1.379880e+00], [-2.634770e+00, 1.749020e+00, 1.197270e+00, -1.339840e+00], [8.872070e-01, -1.368160e+00, -3.212890e+00, -1.324220e+00], [3.062500e+00, 3.812500e+00, 3.285160e+00, -3.181640e+00], [-8.001710e-02, -6.078130e+00, 1.148440e+00, -4.992190e+00]]]> : tensor<3x5x4xf16>
    %1 = stablehlo.constant dense<[[[3.873050e+00, 1.847660e+00, -6.606450e-01, 3.136720e+00], [-4.089840e+00, -2.451170e+00, -1.253910e+00, -6.430660e-01]], [[1.850590e+00, -1.354490e+00, -1.544920e+00, 2.496340e-01], [4.401860e-01, 1.892580e+00, 1.561520e+00, -2.880860e+00]], [[9.145500e-01, -6.958000e-01, 5.226560e+00, 4.386720e+00], [1.200560e-01, 3.513670e+00, 5.390630e+00, 7.221670e-01]]]> : tensor<3x2x4xf16>
    return %0, %1 : tensor<3x5x4xf16>, tensor<3x2x4xf16>
  }
  func.func private @expected() -> tensor<3x5x4xf16> {
    %0 = stablehlo.constant dense<[[[-2.097660e+00, -1.900390e+00, -9.916990e-01, -6.335940e+00], [5.012500e+01, -1.884380e+01, 2.812500e-01, -4.804690e+00], [-2.152340e+00, -4.033200e-01, -3.355470e+00, 1.889650e+00], [-4.093750e+00, 1.389650e+00, -3.257810e+00, 2.123050e+00], [3.828130e-01, 1.972660e+00, -1.303710e+00, 1.476560e+00]], [[3.105470e+00, -1.161130e+00, -5.024410e-01, -5.316410e+00], [-3.531250e+00, -7.328130e+00, -1.385940e+01, -7.333980e-01], [-4.214840e+00, -1.667970e+00, -2.269530e+00, -5.812500e+00], [4.072270e-01, -4.964840e+00, 3.513180e-01, -2.195310e+00], [5.844730e-01, -4.820310e+00, -8.486330e-01, 4.093750e+00]], [[-3.029790e-01, -4.013670e-01, -3.593750e+00, -1.379880e+00], [-2.893070e-01, -4.277340e+00, 3.371880e+01, -4.246090e+00], [8.872070e-01, -1.368160e+00, -3.212890e+00, -1.324220e+00], [3.062500e+00, 3.812500e+00, 3.285160e+00, -3.181640e+00], [-8.001710e-02, -6.078130e+00, 1.148440e+00, -4.992190e+00]]]> : tensor<3x5x4xf16>
    return %0 : tensor<3x5x4xf16>
  }
}

