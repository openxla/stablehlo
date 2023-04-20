// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xbf16>, tensor<7x4xbf16>)
    %1 = call @expected() : () -> tensor<7x3xbf16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<7x3x4xbf16>, tensor<7x4xbf16>) -> tensor<7x3xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xbf16>, tensor<7x3xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xbf16>, tensor<7x4xbf16>) {
    %0 = stablehlo.constant dense<[[[1.062500e+00, -6.914060e-01, -4.937500e+00, 3.000000e+00], [1.708980e-01, -3.828130e+00, 2.265630e+00, -1.070310e+00], [-2.906250e+00, -3.703130e+00, 1.984380e+00, -1.335940e+00]], [[7.304680e-01, 6.000000e+00, 2.062500e+00, 2.187500e+00], [3.750000e+00, -4.375000e+00, -3.796880e+00, 4.736330e-02], [-1.640630e+00, 1.476560e+00, -1.386720e-01, 2.968750e-01]], [[-2.375000e+00, -2.406250e+00, -1.773440e+00, 2.593750e+00], [7.148430e-01, -5.859380e-01, -2.156250e+00, -4.628910e-01], [-7.031250e-01, -2.046880e+00, -1.812500e+00, 4.593750e+00]], [[-1.023440e+00, 5.117190e-01, -1.289060e+00, -5.062500e+00], [-1.375000e+00, 1.648440e+00, -5.875000e+00, -4.781250e+00], [1.093750e+00, -5.656250e+00, -2.359380e+00, 4.707030e-01]], [[-3.921880e+00, -9.843750e-01, -1.929690e+00, 1.546880e+00], [6.500000e+00, -6.750000e+00, 4.656250e+00, 4.750000e+00], [-5.718750e+00, -4.125000e+00, -1.382810e+00, 1.351560e+00]], [[4.355470e-01, 4.238280e-01, -2.558590e-01, 5.343750e+00], [-1.445310e-01, 3.671880e+00, 5.507810e-01, 4.187500e+00], [1.578130e+00, -3.171880e+00, -1.648440e+00, 4.156250e+00]], [[2.296880e+00, 6.750000e+00, 1.398440e+00, -1.031250e+00], [2.234380e+00, 4.093750e+00, -1.148440e+00, 1.039060e+00], [5.820310e-01, 2.390630e+00, -2.546880e+00, 1.718750e+00]]]> : tensor<7x3x4xbf16>
    %1 = stablehlo.constant dense<[[-2.304690e-01, -4.125000e+00, 3.984380e+00, -6.343750e+00], [-3.875000e+00, -5.531250e+00, 2.375000e+00, -3.062500e+00], [-3.765630e+00, -7.421880e-01, -2.031250e+00, 8.789060e-01], [-1.781250e+00, -4.593750e+00, -5.195310e-01, -8.359380e-01], [-1.621090e-01, -5.390630e-01, 4.550780e-01, -8.671870e-01], [-2.218750e+00, -1.672360e-02, -3.593750e-01, -7.000000e+00], [-5.531250e+00, 9.882810e-01, -1.187500e+00, -2.734380e+00]]> : tensor<7x4xbf16>
    return %0, %1 : tensor<7x3x4xbf16>, tensor<7x4xbf16>
  }
  func.func private @expected() -> tensor<7x3xbf16> {
    %0 = stablehlo.constant dense<[[-3.600000e+01, 3.162500e+01, 3.225000e+01], [-3.775000e+01, 5.039060e-01, -3.046880e+00], [1.662500e+01, 1.718750e+00, 1.187500e+01], [4.375000e+00, 1.921880e+00, 2.487500e+01], [-1.054690e+00, 5.859380e-01, 1.351560e+00], [-3.825000e+01, -2.925000e+01, -3.200000e+01], [-4.875000e+00, -9.812500e+00, -2.531250e+00]]> : tensor<7x3xbf16>
    return %0 : tensor<7x3xbf16>
  }
}

