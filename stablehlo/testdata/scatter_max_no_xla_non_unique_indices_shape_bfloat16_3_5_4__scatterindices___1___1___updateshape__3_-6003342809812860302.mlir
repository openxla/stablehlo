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
    %0 = stablehlo.constant dense<[[[-2.062500e+00, 6.679690e-01, 3.734380e+00, -1.492190e+00], [3.265630e+00, 4.875000e+00, -1.578130e+00, -3.921880e+00], [-1.030270e-01, 4.468750e+00, -4.750000e+00, -4.968750e+00], [-1.296880e+00, -3.375000e+00, 4.062500e+00, -2.390630e+00], [-1.765630e+00, -2.718750e+00, 1.816410e-01, 1.320310e+00]], [[-8.437500e-01, 7.968750e-01, -6.445310e-01, 2.470700e-01], [1.148440e+00, 2.312500e+00, 6.125000e+00, 4.625000e+00], [3.578130e+00, 9.609370e-01, 3.218750e+00, -1.455080e-01], [3.031250e+00, -9.804680e-01, 3.218750e+00, -3.875000e+00], [-3.007810e-01, 2.937500e+00, 5.718750e+00, 1.492190e+00]], [[3.859380e+00, -2.046880e+00, 4.687500e-01, 1.789060e+00], [1.281250e+00, -9.414060e-01, -2.796880e+00, -4.156250e+00], [-7.148430e-01, 2.484380e+00, -2.125000e+00, -1.929690e+00], [1.562500e+00, 1.648440e+00, 4.375000e+00, -7.226560e-01], [5.351560e-01, 1.898440e+00, 3.296880e+00, 5.898440e-01]]]> : tensor<3x5x4xbf16>
    %1 = stablehlo.constant dense<[[[5.937500e-01, -2.625000e+00, -3.093750e+00, -3.781250e+00], [-4.218750e+00, 4.531250e+00, -4.921880e-01, 4.343750e+00]], [[1.351560e+00, -7.773430e-01, 2.500000e+00, 4.062500e+00], [7.773430e-01, 1.656250e+00, 1.320310e+00, 6.875000e+00]], [[1.875000e+00, 5.906250e+00, 1.070310e+00, -3.046880e+00], [2.375000e+00, 6.015630e-01, 1.640630e+00, 3.843750e+00]]]> : tensor<3x2x4xbf16>
    return %0, %1 : tensor<3x5x4xbf16>, tensor<3x2x4xbf16>
  }
  func.func private @expected() -> tensor<3x5x4xbf16> {
    %0 = stablehlo.constant dense<[[[-2.062500e+00, 6.679690e-01, 3.734380e+00, -1.492190e+00], [3.265630e+00, 4.875000e+00, -4.921880e-01, 4.343750e+00], [-1.030270e-01, 4.468750e+00, -4.750000e+00, -4.968750e+00], [-1.296880e+00, -3.375000e+00, 4.062500e+00, -2.390630e+00], [-1.765630e+00, -2.718750e+00, 1.816410e-01, 1.320310e+00]], [[-8.437500e-01, 7.968750e-01, -6.445310e-01, 2.470700e-01], [1.351560e+00, 2.312500e+00, 6.125000e+00, 6.875000e+00], [3.578130e+00, 9.609370e-01, 3.218750e+00, -1.455080e-01], [3.031250e+00, -9.804680e-01, 3.218750e+00, -3.875000e+00], [-3.007810e-01, 2.937500e+00, 5.718750e+00, 1.492190e+00]], [[3.859380e+00, -2.046880e+00, 4.687500e-01, 1.789060e+00], [2.375000e+00, 5.906250e+00, 1.640630e+00, 3.843750e+00], [-7.148430e-01, 2.484380e+00, -2.125000e+00, -1.929690e+00], [1.562500e+00, 1.648440e+00, 4.375000e+00, -7.226560e-01], [5.351560e-01, 1.898440e+00, 3.296880e+00, 5.898440e-01]]]> : tensor<3x5x4xbf16>
    return %0 : tensor<3x5x4xbf16>
  }
}

