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
    %0 = stablehlo.constant dense<[[[2.824710e-01, -1.199220e+00, 4.375000e+00, -4.820310e+00], [-2.302730e+00, 4.169920e-01, -1.762700e+00, -3.402340e+00], [1.759770e+00, -1.086910e+00, -1.688480e+00, -2.994140e+00], [-3.031250e+00, 1.537110e+00, -7.167960e-01, 6.550780e+00], [-4.386720e+00, -3.103520e+00, -3.281250e+00, -8.793940e-01]], [[-4.377440e-01, -3.865230e+00, 3.964840e+00, 1.154300e+00], [1.540040e+00, -1.220700e+00, 2.751460e-01, -3.072270e+00], [-1.175780e+00, 8.212890e-01, 5.207030e+00, -2.097660e+00], [-5.960940e+00, 5.691400e+00, 1.075200e+00, -3.082030e+00], [2.207030e+00, -2.128910e+00, 2.541020e+00, 1.536130e+00]], [[2.927730e+00, -1.592770e+00, 3.929690e+00, -6.015630e+00], [-8.496090e-01, -9.560540e-01, 2.623050e+00, -2.250000e+00], [1.618160e+00, 1.528320e+00, -1.177730e+00, 1.017580e+00], [7.426750e-01, 5.699210e+00, 2.003910e+00, 2.849610e+00], [4.160160e-01, 2.500000e+00, -4.262700e-01, 2.308590e+00]]]> : tensor<3x5x4xf16>
    %1 = stablehlo.constant dense<[[[3.710940e+00, 3.078130e+00, -4.074220e+00, 1.054690e+00], [-1.595700e+00, -2.458980e+00, -1.033200e+00, 4.128910e+00]], [[7.602530e-01, 1.813480e+00, 3.976560e+00, 3.130860e+00], [8.427730e-01, 2.388670e+00, 6.156250e+00, -3.082030e+00]], [[-2.798830e+00, -3.732420e+00, 5.581050e-01, 6.792960e+00], [1.000000e+00, 4.406250e+00, -4.175780e+00, -1.073240e+00]]]> : tensor<3x2x4xf16>
    return %0, %1 : tensor<3x5x4xf16>, tensor<3x2x4xf16>
  }
  func.func private @expected() -> tensor<3x5x4xf16> {
    %0 = stablehlo.constant dense<[[[2.824710e-01, -1.199220e+00, 4.375000e+00, -4.820310e+00], [1.364060e+01, -3.156250e+00, -7.417960e+00, -1.481250e+01], [1.759770e+00, -1.086910e+00, -1.688480e+00, -2.994140e+00], [-3.031250e+00, 1.537110e+00, -7.167960e-01, 6.550780e+00], [-4.386720e+00, -3.103520e+00, -3.281250e+00, -8.793940e-01]], [[-4.377440e-01, -3.865230e+00, 3.964840e+00, 1.154300e+00], [9.868160e-01, -5.285160e+00, 6.734380e+00, 2.964060e+01], [-1.175780e+00, 8.212890e-01, 5.207030e+00, -2.097660e+00], [-5.960940e+00, 5.691400e+00, 1.075200e+00, -3.082030e+00], [2.207030e+00, -2.128910e+00, 2.541020e+00, 1.536130e+00]], [[2.927730e+00, -1.592770e+00, 3.929690e+00, -6.015630e+00], [2.376950e+00, 1.572660e+01, -6.113280e+00, 1.640630e+01], [1.618160e+00, 1.528320e+00, -1.177730e+00, 1.017580e+00], [7.426750e-01, 5.699210e+00, 2.003910e+00, 2.849610e+00], [4.160160e-01, 2.500000e+00, -4.262700e-01, 2.308590e+00]]]> : tensor<3x5x4xf16>
    return %0 : tensor<3x5x4xf16>
  }
}

