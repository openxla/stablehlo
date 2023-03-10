// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>)
    %2 = call @expected() : () -> tensor<3x5x4xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xbf16>, tensor<2x1xi32>, tensor<3x2x4xbf16>) -> tensor<3x5x4xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xbf16>, tensor<3x5x4xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>) {
    %0 = stablehlo.constant dense<[[[-3.390630e+00, -1.820310e+00, 3.859380e+00, -2.703130e+00], [1.718750e+00, 3.890630e+00, 5.000000e+00, 4.156250e+00], [-7.343750e-01, -2.437500e+00, -1.343750e+00, -1.742190e+00], [1.765630e+00, -1.640630e+00, 4.257810e-01, -5.531250e+00], [1.945310e+00, 5.187500e+00, 6.187500e+00, -1.320310e+00]], [[-1.031250e+00, 1.250000e+00, 2.929690e-01, -2.158200e-01], [1.962890e-01, -2.828130e+00, -5.500000e+00, 1.382810e+00], [4.062500e-01, 7.226560e-02, -4.781250e+00, 1.398440e+00], [-2.656250e-01, 7.578130e-01, -1.960940e+00, 4.593750e+00], [3.906250e+00, 4.906250e+00, -5.000000e+00, -5.031250e+00]], [[-5.703130e-01, -2.546880e+00, 5.968750e+00, 2.671880e+00], [-2.515630e+00, 1.382810e+00, -6.328130e-01, -2.871090e-01], [2.265630e+00, 6.601560e-01, 1.500000e+00, 2.296880e+00], [1.867190e+00, -5.625000e+00, 2.046880e+00, -4.281250e+00], [1.265630e+00, -5.062500e+00, 5.375000e+00, 3.421880e+00]]]> : tensor<3x5x4xbf16>
    %1 = stablehlo.constant dense<[[[-3.390630e+00, 3.734380e+00, -1.781250e+00, -3.171880e+00], [1.656250e+00, -1.609380e+00, -1.828130e+00, 9.960930e-01]], [[-1.601560e+00, -5.031250e+00, 4.218750e+00, 4.750000e+00], [7.929680e-01, -3.535160e-01, -8.046880e-01, 6.906250e+00]], [[-1.398440e+00, -8.437500e+00, 6.445310e-01, -2.910160e-01], [-4.656250e+00, -6.406250e-01, 4.980470e-01, 1.664060e+00]]]> : tensor<3x2x4xbf16>
    return %0, %1 : tensor<3x5x4xbf16>, tensor<3x2x4xbf16>
  }
  func.func private @expected() -> tensor<3x5x4xbf16> {
    %0 = stablehlo.constant dense<[[[-3.390630e+00, -1.820310e+00, 3.859380e+00, -2.703130e+00], [-1.562500e-02, 6.000000e+00, 1.390630e+00, 1.984380e+00], [-7.343750e-01, -2.437500e+00, -1.343750e+00, -1.742190e+00], [1.765630e+00, -1.640630e+00, 4.257810e-01, -5.531250e+00], [1.945310e+00, 5.187500e+00, 6.187500e+00, -1.320310e+00]], [[-1.031250e+00, 1.250000e+00, 2.929690e-01, -2.158200e-01], [-6.132810e-01, -8.250000e+00, -2.093750e+00, 1.300000e+01], [4.062500e-01, 7.226560e-02, -4.781250e+00, 1.398440e+00], [-2.656250e-01, 7.578130e-01, -1.960940e+00, 4.593750e+00], [3.906250e+00, 4.906250e+00, -5.000000e+00, -5.031250e+00]], [[-5.703130e-01, -2.546880e+00, 5.968750e+00, 2.671880e+00], [-8.562500e+00, -7.687500e+00, 5.078130e-01, 1.085940e+00], [2.265630e+00, 6.601560e-01, 1.500000e+00, 2.296880e+00], [1.867190e+00, -5.625000e+00, 2.046880e+00, -4.281250e+00], [1.265630e+00, -5.062500e+00, 5.375000e+00, 3.421880e+00]]]> : tensor<3x5x4xbf16>
    return %0 : tensor<3x5x4xbf16>
  }
}

