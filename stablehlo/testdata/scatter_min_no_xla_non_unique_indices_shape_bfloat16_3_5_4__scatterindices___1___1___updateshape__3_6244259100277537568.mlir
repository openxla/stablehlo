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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xbf16>, tensor<2x1xi32>, tensor<3x2x4xbf16>) -> tensor<3x5x4xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xbf16>, tensor<3x5x4xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>) {
    %0 = stablehlo.constant dense<[[[-8.007810e-01, 5.062500e+00, 1.031250e+00, -3.718750e+00], [-2.015630e+00, -1.750000e+00, -2.226560e-01, 2.593750e+00], [-2.984380e+00, 3.359380e+00, -1.552730e-01, -2.281250e+00], [5.117190e-01, -5.968750e+00, -2.328130e+00, -4.968750e+00], [-2.773440e-01, -3.875000e+00, 2.562500e+00, 1.539060e+00]], [[7.500000e-01, -5.468750e+00, 6.125000e+00, 4.638670e-02], [1.398440e+00, -7.148430e-01, -1.218750e+00, -1.359380e+00], [-1.101560e+00, -3.484380e+00, -4.281250e+00, 7.578130e-01], [2.453130e+00, 5.343750e+00, 3.984380e+00, -2.734380e+00], [-2.625000e+00, -3.656250e+00, -6.127930e-02, 6.437500e+00]], [[3.515630e+00, -1.101560e+00, 4.656250e+00, 4.593750e+00], [6.054690e-01, -9.414060e-01, 2.125000e+00, -2.812500e+00], [1.406250e+00, -1.335940e+00, 4.589840e-01, -4.437500e+00], [-1.781250e+00, -7.539060e-01, -1.234380e+00, 6.562500e-01], [-1.203130e+00, -1.242190e+00, 2.562500e+00, -9.609370e-01]]]> : tensor<3x5x4xbf16>
    %1 = stablehlo.constant dense<[[[4.187500e+00, 1.140630e+00, -7.656250e-01, 3.281250e+00], [3.750000e-01, 1.718750e+00, -5.234380e-01, 1.656250e+00]], [[2.656250e-01, -1.281250e+00, 8.125000e+00, -2.968750e+00], [-3.796880e+00, -2.968750e+00, -9.882810e-01, -5.273440e-01]], [[-2.531250e+00, 2.734380e+00, -3.468750e+00, -3.406250e+00], [3.000000e+00, -8.085930e-01, -2.218750e+00, 1.125000e+00]]]> : tensor<3x2x4xbf16>
    return %0, %1 : tensor<3x5x4xbf16>, tensor<3x2x4xbf16>
  }
  func.func private @expected() -> tensor<3x5x4xbf16> {
    %0 = stablehlo.constant dense<[[[-8.007810e-01, 5.062500e+00, 1.031250e+00, -3.718750e+00], [-2.015630e+00, -1.750000e+00, -7.656250e-01, 1.656250e+00], [-2.984380e+00, 3.359380e+00, -1.552730e-01, -2.281250e+00], [5.117190e-01, -5.968750e+00, -2.328130e+00, -4.968750e+00], [-2.773440e-01, -3.875000e+00, 2.562500e+00, 1.539060e+00]], [[7.500000e-01, -5.468750e+00, 6.125000e+00, 4.638670e-02], [-3.796880e+00, -2.968750e+00, -1.218750e+00, -2.968750e+00], [-1.101560e+00, -3.484380e+00, -4.281250e+00, 7.578130e-01], [2.453130e+00, 5.343750e+00, 3.984380e+00, -2.734380e+00], [-2.625000e+00, -3.656250e+00, -6.127930e-02, 6.437500e+00]], [[3.515630e+00, -1.101560e+00, 4.656250e+00, 4.593750e+00], [-2.531250e+00, -9.414060e-01, -3.468750e+00, -3.406250e+00], [1.406250e+00, -1.335940e+00, 4.589840e-01, -4.437500e+00], [-1.781250e+00, -7.539060e-01, -1.234380e+00, 6.562500e-01], [-1.203130e+00, -1.242190e+00, 2.562500e+00, -9.609370e-01]]]> : tensor<3x5x4xbf16>
    return %0 : tensor<3x5x4xbf16>
  }
}

