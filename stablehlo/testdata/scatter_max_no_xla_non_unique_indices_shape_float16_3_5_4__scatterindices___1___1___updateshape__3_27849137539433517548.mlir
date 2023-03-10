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
    %0 = stablehlo.constant dense<[[[-4.992190e+00, -1.767580e+00, -3.073730e-01, 2.175780e+00], [9.931640e-01, 2.708980e+00, 3.634770e+00, 3.138670e+00], [4.480470e+00, 3.230470e+00, -1.224610e+00, -3.285160e+00], [4.015630e+00, -1.343750e+00, 1.248050e+00, 7.119140e-01], [-6.767580e-01, 3.151860e-01, -1.113280e+00, -9.760740e-01]], [[1.958980e+00, -6.997070e-01, 8.569330e-01, -1.395510e+00], [-2.427730e+00, -1.132810e+00, -3.383790e-01, -2.554690e+00], [1.437500e+00, 3.939450e+00, 5.562500e+00, 2.212890e+00], [2.042970e+00, 3.618160e-01, -1.085940e+00, -5.083010e-01], [-2.964840e+00, -1.817380e+00, 3.861330e+00, -1.304690e+00]], [[-1.211910e+00, 2.142580e+00, 3.435550e+00, -6.671880e+00], [-4.292970e+00, -4.628910e+00, 5.343750e+00, 1.074220e+00], [9.804680e-01, 1.466800e+00, -7.523430e+00, -6.820310e+00], [3.363280e+00, -4.269530e+00, -3.148440e+00, -3.080080e+00], [-8.364250e-01, 3.914060e+00, 2.162110e+00, -2.841800e+00]]]> : tensor<3x5x4xf16>
    %1 = stablehlo.constant dense<[[[4.921880e+00, 2.158200e+00, 1.696290e+00, -4.414060e+00], [9.741210e-01, 2.941410e+00, -2.644040e-01, -1.743160e-01]], [[-3.919920e+00, -2.597660e+00, 3.632810e+00, -6.207030e+00], [2.001950e-01, 5.214840e-01, 2.621090e+00, 8.935540e-01]], [[-2.119140e+00, 4.808590e+00, -4.246090e+00, 5.378900e+00], [-3.089840e+00, 3.800780e+00, 2.478520e+00, -4.656250e+00]]]> : tensor<3x2x4xf16>
    return %0, %1 : tensor<3x5x4xf16>, tensor<3x2x4xf16>
  }
  func.func private @expected() -> tensor<3x5x4xf16> {
    %0 = stablehlo.constant dense<[[[-4.992190e+00, -1.767580e+00, -3.073730e-01, 2.175780e+00], [4.921880e+00, 2.941410e+00, 3.634770e+00, 3.138670e+00], [4.480470e+00, 3.230470e+00, -1.224610e+00, -3.285160e+00], [4.015630e+00, -1.343750e+00, 1.248050e+00, 7.119140e-01], [-6.767580e-01, 3.151860e-01, -1.113280e+00, -9.760740e-01]], [[1.958980e+00, -6.997070e-01, 8.569330e-01, -1.395510e+00], [2.001950e-01, 5.214840e-01, 3.632810e+00, 8.935540e-01], [1.437500e+00, 3.939450e+00, 5.562500e+00, 2.212890e+00], [2.042970e+00, 3.618160e-01, -1.085940e+00, -5.083010e-01], [-2.964840e+00, -1.817380e+00, 3.861330e+00, -1.304690e+00]], [[-1.211910e+00, 2.142580e+00, 3.435550e+00, -6.671880e+00], [-2.119140e+00, 4.808590e+00, 5.343750e+00, 5.378900e+00], [9.804680e-01, 1.466800e+00, -7.523430e+00, -6.820310e+00], [3.363280e+00, -4.269530e+00, -3.148440e+00, -3.080080e+00], [-8.364250e-01, 3.914060e+00, 2.162110e+00, -2.841800e+00]]]> : tensor<3x5x4xf16>
    return %0 : tensor<3x5x4xf16>
  }
}

