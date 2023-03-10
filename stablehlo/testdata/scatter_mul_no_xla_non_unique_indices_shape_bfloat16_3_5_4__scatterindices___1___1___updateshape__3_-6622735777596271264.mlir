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
    %0 = stablehlo.constant dense<[[[-7.070310e-01, -1.335940e+00, -7.109380e-01, 8.906250e-01], [3.281250e+00, 1.040040e-01, -3.066410e-01, -4.343750e+00], [1.218750e+00, 1.937500e+00, 2.636720e-01, 4.437500e+00], [1.093750e+00, -3.312500e+00, -5.156250e-01, -4.500000e+00], [-6.218750e+00, 1.484380e+00, 2.890630e-01, -4.343750e+00]], [[-1.117190e+00, -2.296880e+00, -3.765630e+00, -1.031250e+00], [3.078130e+00, -2.392580e-01, -3.015630e+00, 4.609380e-01], [-1.492190e+00, -5.039060e-01, -1.585940e+00, 3.046880e+00], [1.226560e+00, 2.197270e-01, -3.765630e+00, -2.312500e+00], [-3.750000e+00, 3.609380e+00, 1.539060e+00, 2.859380e+00]], [[9.218750e-01, 4.968750e+00, 1.593750e+00, -2.625000e+00], [-3.671880e+00, -2.953130e+00, -1.171880e+00, 1.140630e+00], [-2.328130e+00, -1.484380e+00, 4.687500e+00, -5.000000e+00], [3.945310e-01, -2.470700e-01, -2.109380e+00, 1.406250e+00], [2.765630e+00, -8.476560e-01, -4.000000e+00, 9.687500e-01]]]> : tensor<3x5x4xbf16>
    %1 = stablehlo.constant dense<[[[-2.718750e+00, -4.468750e+00, -3.906250e+00, 2.687500e+00], [6.289060e-01, -8.625000e+00, 6.906250e+00, 1.101560e+00]], [[2.656250e+00, -1.960940e+00, 4.250000e+00, 2.578130e+00], [-2.093750e+00, -7.109380e-01, 3.671880e+00, -5.468750e-01]], [[3.078130e+00, -1.671880e+00, -2.187500e+00, -5.273440e-01], [5.156250e+00, -2.890630e-01, 1.078130e+00, -3.703130e+00]]]> : tensor<3x2x4xbf16>
    return %0, %1 : tensor<3x5x4xbf16>, tensor<3x2x4xbf16>
  }
  func.func private @expected() -> tensor<3x5x4xbf16> {
    %0 = stablehlo.constant dense<[[[-7.070310e-01, -1.335940e+00, -7.109380e-01, 8.906250e-01], [-5.625000e+00, 4.000000e+00, 8.250000e+00, -1.287500e+01], [1.218750e+00, 1.937500e+00, 2.636720e-01, 4.437500e+00], [1.093750e+00, -3.312500e+00, -5.156250e-01, -4.500000e+00], [-6.218750e+00, 1.484380e+00, 2.890630e-01, -4.343750e+00]], [[-1.117190e+00, -2.296880e+00, -3.765630e+00, -1.031250e+00], [-1.712500e+01, -3.339840e-01, -4.700000e+01, -6.484380e-01], [-1.492190e+00, -5.039060e-01, -1.585940e+00, 3.046880e+00], [1.226560e+00, 2.197270e-01, -3.765630e+00, -2.312500e+00], [-3.750000e+00, 3.609380e+00, 1.539060e+00, 2.859380e+00]], [[9.218750e-01, 4.968750e+00, 1.593750e+00, -2.625000e+00], [-5.825000e+01, -1.429690e+00, 2.765630e+00, 2.234380e+00], [-2.328130e+00, -1.484380e+00, 4.687500e+00, -5.000000e+00], [3.945310e-01, -2.470700e-01, -2.109380e+00, 1.406250e+00], [2.765630e+00, -8.476560e-01, -4.000000e+00, 9.687500e-01]]]> : tensor<3x5x4xbf16>
    return %0 : tensor<3x5x4xbf16>
  }
}

