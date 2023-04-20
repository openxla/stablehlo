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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xbf16>, tensor<2x1xi32>, tensor<3x2x4xbf16>) -> tensor<3x5x4xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xbf16>, tensor<3x5x4xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>) {
    %0 = stablehlo.constant dense<[[[-4.156250e+00, -9.570310e-01, 1.648440e+00, 4.343750e+00], [5.468750e-01, 2.687500e+00, 2.031250e+00, 1.781250e+00], [-7.968750e-01, 1.191410e-01, 8.867180e-01, -5.937500e-01], [2.468750e+00, 6.914060e-01, -4.281250e+00, 2.871090e-01], [-8.750000e-01, -1.281250e+00, -1.148440e+00, -5.585940e-01]], [[1.953130e+00, -3.343750e+00, 1.734380e+00, -3.265630e+00], [-1.468750e+00, -6.093750e+00, -3.453130e+00, -1.546880e+00], [-1.906250e+00, 1.773440e+00, -4.355470e-01, -7.812500e-01], [3.656250e+00, -2.431640e-01, -1.296880e+00, 2.421880e+00], [-2.703130e+00, -2.875000e+00, 8.125000e+00, 3.281250e+00]], [[3.265630e+00, 1.734380e+00, -3.125000e+00, 8.750000e-01], [-1.406250e+00, -1.320310e+00, 1.318360e-01, 5.195310e-01], [-1.398440e+00, -1.742190e+00, 4.343750e+00, -3.656250e+00], [-3.687500e+00, 1.328130e+00, -3.500000e+00, -6.132810e-01], [-1.515630e+00, 1.601560e-01, 1.976560e+00, 3.109380e+00]]]> : tensor<3x5x4xbf16>
    %1 = stablehlo.constant dense<[[[-2.500000e+00, 2.484380e+00, -1.000980e-01, -5.500000e+00], [-5.687500e+00, -1.617190e+00, -2.156250e+00, -1.179690e+00]], [[-1.460940e+00, 2.203130e+00, 2.546880e+00, -2.656250e+00], [5.687500e+00, 4.187500e+00, -1.562500e+00, 1.476560e+00]], [[3.000000e+00, 1.671880e+00, -5.593750e+00, 3.484380e+00], [-1.296880e+00, -1.215820e-01, 1.000000e+01, -7.187500e-01]]]> : tensor<3x2x4xbf16>
    return %0, %1 : tensor<3x5x4xbf16>, tensor<3x2x4xbf16>
  }
  func.func private @expected() -> tensor<3x5x4xbf16> {
    %0 = stablehlo.constant dense<[[[-4.156250e+00, -9.570310e-01, 1.648440e+00, 4.343750e+00], [5.468750e-01, 2.687500e+00, 2.031250e+00, 1.781250e+00], [-7.968750e-01, 1.191410e-01, 8.867180e-01, -5.937500e-01], [2.468750e+00, 6.914060e-01, -4.281250e+00, 2.871090e-01], [-8.750000e-01, -1.281250e+00, -1.148440e+00, -5.585940e-01]], [[1.953130e+00, -3.343750e+00, 1.734380e+00, -3.265630e+00], [5.687500e+00, 4.187500e+00, 2.546880e+00, 1.476560e+00], [-1.906250e+00, 1.773440e+00, -4.355470e-01, -7.812500e-01], [3.656250e+00, -2.431640e-01, -1.296880e+00, 2.421880e+00], [-2.703130e+00, -2.875000e+00, 8.125000e+00, 3.281250e+00]], [[3.265630e+00, 1.734380e+00, -3.125000e+00, 8.750000e-01], [3.000000e+00, 1.671880e+00, 1.000000e+01, 3.484380e+00], [-1.398440e+00, -1.742190e+00, 4.343750e+00, -3.656250e+00], [-3.687500e+00, 1.328130e+00, -3.500000e+00, -6.132810e-01], [-1.515630e+00, 1.601560e-01, 1.976560e+00, 3.109380e+00]]]> : tensor<3x5x4xbf16>
    return %0 : tensor<3x5x4xbf16>
  }
}

