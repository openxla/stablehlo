// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xbf16>, tensor<7x4xbf16>)
    %1 = call @expected() : () -> tensor<7x3xbf16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<7x3x4xbf16>, tensor<7x4xbf16>) -> tensor<7x3xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xbf16>, tensor<7x3xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xbf16>, tensor<7x4xbf16>) {
    %0 = stablehlo.constant dense<[[[-7.695310e-01, 1.257810e+00, -2.984380e+00, 8.046880e-01], [2.781250e+00, 2.281250e+00, -4.625000e+00, -2.593750e+00], [-1.234380e+00, -1.226560e+00, -4.781250e+00, -6.787110e-02]], [[-1.000000e+00, -3.109380e+00, -2.015630e+00, 9.882810e-01], [3.457030e-01, -5.859380e-01, -6.093750e-01, 2.359380e+00], [-4.902340e-01, 6.406250e+00, -1.578130e+00, 4.218750e+00]], [[-3.328130e+00, 4.906250e+00, 2.484380e+00, -1.601560e-01], [7.875000e+00, 3.281250e+00, 3.281250e+00, -2.828130e+00], [1.992190e+00, -3.718750e+00, 5.703130e-01, -1.123050e-01]], [[1.484380e+00, -7.773430e-01, -2.062500e+00, 9.414060e-01], [-1.359380e+00, -1.109380e+00, -3.710940e-01, -1.804690e+00], [-2.140630e+00, -2.156250e+00, 2.412110e-01, -6.406250e+00]], [[-4.375000e+00, -1.687500e+00, 1.484380e+00, -3.781250e+00], [6.812500e+00, 1.445310e+00, 1.796880e+00, 6.156250e+00], [-1.851560e+00, -1.718750e+00, 1.164060e+00, -2.625000e+00]], [[-6.757810e-01, -1.148440e+00, -2.734380e+00, 2.750000e+00], [-2.125000e+00, 6.843750e+00, 1.101560e+00, 4.125000e+00], [-3.442380e-02, 9.619140e-02, 5.390630e-01, -1.585940e+00]], [[3.093750e+00, -5.968750e+00, 5.195310e-01, -2.171880e+00], [-3.203130e+00, -1.765630e+00, -2.578130e-01, 2.546880e+00], [-4.281250e+00, 6.367190e-01, -4.468750e+00, 3.609380e+00]]]> : tensor<7x3x4xbf16>
    %1 = stablehlo.constant dense<[[2.656250e+00, 4.718750e+00, -5.898440e-01, 1.664060e+00], [2.953130e+00, 5.281250e+00, 3.031250e+00, 1.507810e+00], [-6.796880e-01, -2.875000e+00, 9.082030e-02, -3.247070e-02], [4.121090e-01, -7.531250e+00, -4.250000e+00, 6.796880e-01], [3.937500e+00, -1.281250e+00, -3.710940e-02, -3.296880e+00], [2.656250e+00, 1.398440e+00, 7.718750e+00, 1.257810e+00], [-4.218750e+00, -4.218750e+00, -4.687500e+00, 1.820310e+00]]> : tensor<7x4xbf16>
    return %0, %1 : tensor<7x3x4xbf16>, tensor<7x4xbf16>
  }
  func.func private @expected() -> tensor<7x3xbf16> {
    %0 = stablehlo.constant dense<[[7.000000e+00, 1.662500e+01, -6.375000e+00], [-2.400000e+01, -3.632810e-01, 3.400000e+01], [-1.162500e+01, -1.437500e+01, 9.375000e+00], [1.587500e+01, 8.125000e+00, 1.000000e+01], [-2.656250e+00, 4.593750e+00, 3.515630e+00], [-2.100000e+01, 1.762500e+01, 2.203130e+00], [5.750000e+00, 2.675000e+01, 4.300000e+01]]> : tensor<7x3xbf16>
    return %0 : tensor<7x3xbf16>
  }
}
