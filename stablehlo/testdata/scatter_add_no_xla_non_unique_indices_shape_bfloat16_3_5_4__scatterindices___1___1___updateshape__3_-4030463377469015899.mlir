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
    %0 = stablehlo.constant dense<[[[2.917480e-02, -1.147460e-01, 3.406250e+00, -9.437500e+00], [-1.179690e+00, -9.062500e-01, -2.871090e-01, 5.625000e+00], [-1.312500e+00, -4.468750e+00, 8.046880e-01, 3.031250e+00], [-4.843750e+00, -6.093750e-01, 2.406250e+00, -1.703130e+00], [3.496090e-01, -7.250000e+00, 1.515630e+00, -7.187500e+00]], [[1.414060e+00, 2.671880e+00, -2.093750e+00, 2.109380e+00], [-9.726560e-01, 2.515630e+00, -9.218750e-01, -3.140630e+00], [-3.046880e+00, -4.375000e+00, -3.781250e+00, -9.960930e-01], [-1.593750e+00, 1.039060e+00, 6.210940e-01, -7.125000e+00], [-3.953130e+00, 3.859380e+00, 3.609380e+00, -2.390630e+00]], [[-1.656250e+00, -3.656250e+00, 1.562500e+00, -3.203130e+00], [1.464840e-01, 5.843750e+00, 4.125000e+00, -2.625000e+00], [-5.585940e-01, 5.546880e-01, -6.843750e+00, 3.593750e-01], [6.656250e+00, 2.812500e+00, 3.312500e+00, 8.710930e-01], [-7.812500e-01, 2.109380e+00, -5.437500e+00, -4.750000e+00]]]> : tensor<3x5x4xbf16>
    %1 = stablehlo.constant dense<[[[1.914060e-01, 6.601560e-01, 1.773440e+00, -7.343750e+00], [2.015630e+00, 3.734380e+00, -3.218750e+00, -5.531250e+00]], [[2.359380e+00, -5.187500e+00, -2.812500e+00, -6.992180e-01], [-4.218750e+00, -4.406250e+00, -3.562500e+00, 6.437500e+00]], [[2.312500e+00, -3.187500e+00, 2.792970e-01, -8.789060e-01], [7.187500e-01, -3.015630e+00, 8.125000e-01, -7.128900e-02]]]> : tensor<3x2x4xbf16>
    return %0, %1 : tensor<3x5x4xbf16>, tensor<3x2x4xbf16>
  }
  func.func private @expected() -> tensor<3x5x4xbf16> {
    %0 = stablehlo.constant dense<[[[2.917480e-02, -1.147460e-01, 3.406250e+00, -9.437500e+00], [1.031250e+00, 3.484380e+00, -1.734380e+00, -7.250000e+00], [-1.312500e+00, -4.468750e+00, 8.046880e-01, 3.031250e+00], [-4.843750e+00, -6.093750e-01, 2.406250e+00, -1.703130e+00], [3.496090e-01, -7.250000e+00, 1.515630e+00, -7.187500e+00]], [[1.414060e+00, 2.671880e+00, -2.093750e+00, 2.109380e+00], [-2.828130e+00, -7.062500e+00, -7.312500e+00, 2.593750e+00], [-3.046880e+00, -4.375000e+00, -3.781250e+00, -9.960930e-01], [-1.593750e+00, 1.039060e+00, 6.210940e-01, -7.125000e+00], [-3.953130e+00, 3.859380e+00, 3.609380e+00, -2.390630e+00]], [[-1.656250e+00, -3.656250e+00, 1.562500e+00, -3.203130e+00], [3.171880e+00, -3.593750e-01, 5.218750e+00, -3.578130e+00], [-5.585940e-01, 5.546880e-01, -6.843750e+00, 3.593750e-01], [6.656250e+00, 2.812500e+00, 3.312500e+00, 8.710930e-01], [-7.812500e-01, 2.109380e+00, -5.437500e+00, -4.750000e+00]]]> : tensor<3x5x4xbf16>
    return %0 : tensor<3x5x4xbf16>
  }
}

