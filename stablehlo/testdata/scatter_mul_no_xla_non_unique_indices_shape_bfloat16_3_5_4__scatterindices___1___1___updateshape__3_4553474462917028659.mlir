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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xbf16>, tensor<2x1xi32>, tensor<3x2x4xbf16>) -> tensor<3x5x4xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xbf16>, tensor<3x5x4xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>) {
    %0 = stablehlo.constant dense<[[[5.093750e+00, 2.687500e+00, -4.125000e+00, 1.976560e+00], [1.226560e+00, -1.921880e+00, -2.158200e-01, 3.406250e+00], [6.406250e-01, -3.921880e+00, 7.617180e-01, 3.093750e+00], [-4.121090e-01, 1.789060e+00, -1.533200e-01, -5.859380e-01], [2.453130e+00, 3.183590e-01, 4.625000e+00, -2.812500e+00]], [[-4.250000e+00, 4.625000e+00, 1.265630e+00, 1.507810e+00], [-3.156250e+00, -1.523440e+00, -2.093750e+00, 5.375000e+00], [2.656250e+00, -1.515630e+00, -9.453120e-01, -4.250000e+00], [5.585940e-01, -1.359380e+00, 1.570310e+00, 3.265630e+00], [2.062500e+00, 3.390630e+00, 5.312500e+00, -2.431640e-01]], [[-2.281250e+00, -1.414060e+00, -5.820310e-01, 3.984380e+00], [-2.671880e+00, 5.585940e-01, 1.289060e+00, -4.437500e+00], [2.953130e+00, 9.179680e-01, -1.251220e-02, 3.406250e+00], [1.235350e-01, 2.703130e+00, -5.187500e+00, -7.187500e-01], [-3.484380e+00, -1.898440e+00, 3.015630e+00, 1.953130e+00]]]> : tensor<3x5x4xbf16>
    %1 = stablehlo.constant dense<[[[4.593750e+00, 9.804680e-01, -5.078130e-01, -2.281250e+00], [-3.710940e-01, 4.082030e-01, 3.710940e-01, 1.398440e+00]], [[-6.132810e-01, 4.593750e+00, -8.085930e-01, 3.984380e+00], [-2.203130e+00, -4.125000e+00, -5.039060e-01, 2.609380e+00]], [[-4.843750e+00, 1.687500e+00, -6.176760e-02, 2.968750e-01], [3.328130e+00, 2.890630e+00, 2.093750e+00, -1.664060e+00]]]> : tensor<3x2x4xbf16>
    return %0, %1 : tensor<3x5x4xbf16>, tensor<3x2x4xbf16>
  }
  func.func private @expected() -> tensor<3x5x4xbf16> {
    %0 = stablehlo.constant dense<[[[5.093750e+00, 2.687500e+00, -4.125000e+00, 1.976560e+00], [-2.093750e+00, -7.695310e-01, 4.052730e-02, -1.087500e+01], [6.406250e-01, -3.921880e+00, 7.617180e-01, 3.093750e+00], [-4.121090e-01, 1.789060e+00, -1.533200e-01, -5.859380e-01], [2.453130e+00, 3.183590e-01, 4.625000e+00, -2.812500e+00]], [[-4.250000e+00, 4.625000e+00, 1.265630e+00, 1.507810e+00], [-4.281250e+00, 2.887500e+01, -8.554680e-01, 5.575000e+01], [2.656250e+00, -1.515630e+00, -9.453120e-01, -4.250000e+00], [5.585940e-01, -1.359380e+00, 1.570310e+00, 3.265630e+00], [2.062500e+00, 3.390630e+00, 5.312500e+00, -2.431640e-01]], [[-2.281250e+00, -1.414060e+00, -5.820310e-01, 3.984380e+00], [4.300000e+01, 2.718750e+00, -1.669920e-01, 2.203130e+00], [2.953130e+00, 9.179680e-01, -1.251220e-02, 3.406250e+00], [1.235350e-01, 2.703130e+00, -5.187500e+00, -7.187500e-01], [-3.484380e+00, -1.898440e+00, 3.015630e+00, 1.953130e+00]]]> : tensor<3x5x4xbf16>
    return %0 : tensor<3x5x4xbf16>
  }
}

