// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xbf16>, tensor<7x4xbf16>)
    %1 = call @expected() : () -> tensor<7x3xbf16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<7x3x4xbf16>, tensor<7x4xbf16>) -> tensor<7x3xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xbf16>, tensor<7x3xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xbf16>, tensor<7x4xbf16>) {
    %0 = stablehlo.constant dense<[[[-4.406250e+00, -4.187500e+00, -3.968750e+00, -1.171880e+00], [4.281250e+00, -6.718750e-01, 1.085940e+00, 3.343750e+00], [1.000000e+00, 3.500000e+00, 5.468750e+00, 7.406250e+00]], [[-1.445310e+00, -1.515630e+00, 1.453130e+00, -1.835940e+00], [-4.187500e+00, 5.125000e+00, -3.671880e+00, 2.078130e+00], [-5.234380e-01, -3.531250e+00, -1.312500e+00, 7.531250e+00]], [[-4.406250e+00, 1.421880e+00, -1.523440e+00, 3.078130e+00], [-1.337890e-01, 2.406250e+00, 8.164060e-01, 2.078130e+00], [-8.242180e-01, 2.968750e+00, -3.765630e+00, -4.031250e+00]], [[-2.093750e+00, -2.031250e+00, 4.218750e+00, -3.359380e+00], [1.265630e+00, -1.679690e-01, -3.750000e-01, 1.070310e+00], [-1.265630e+00, 3.218750e+00, -6.982420e-02, 1.046880e+00]], [[2.890630e+00, 1.865230e-01, -1.656250e+00, 7.304680e-01], [-3.578130e+00, 1.117190e+00, 2.695310e-01, -1.789060e+00], [1.406250e+00, -4.593750e+00, -3.140630e+00, 1.765630e+00]], [[-6.156250e+00, 1.656250e+00, -3.343750e+00, -7.890630e-01], [3.687500e+00, 7.250000e+00, 1.157230e-01, 4.656250e+00], [-1.492190e+00, -1.273440e+00, -4.125000e+00, -3.718750e+00]], [[-2.265630e+00, -2.953130e+00, -1.312500e+00, -5.593750e+00], [-2.734380e+00, 3.671880e+00, 7.906250e+00, -5.781250e-01], [-6.531250e+00, 1.953130e+00, 1.601560e+00, -4.937500e+00]]]> : tensor<7x3x4xbf16>
    %1 = stablehlo.constant dense<[[1.929690e+00, 5.500000e+00, 1.429690e+00, -2.875000e+00], [5.468750e+00, 2.171880e+00, -5.898440e-01, 2.531250e+00], [8.359380e-01, 7.275390e-02, 3.578130e+00, 1.359380e+00], [2.312500e+00, 3.015630e+00, 7.812500e-01, -4.394530e-01], [-1.687500e+00, 5.531250e+00, 1.007810e+00, 5.968750e+00], [1.078130e+00, -2.828130e+00, 2.281250e+00, 5.937500e+00], [-2.781250e+00, -2.093750e+00, 4.843750e+00, -5.656250e+00]]> : tensor<7x4xbf16>
    return %0, %1 : tensor<7x3x4xbf16>, tensor<7x4xbf16>
  }
  func.func private @expected() -> tensor<7x3xbf16> {
    %0 = stablehlo.constant dense<[[-3.375000e+01, -3.500000e+00, 7.718750e+00], [-1.675000e+01, -4.343750e+00, 9.312500e+00], [-4.843750e+00, 5.812500e+00, -1.937500e+01], [-6.187500e+00, 1.656250e+00, 6.250000e+00], [-1.156250e+00, 1.812500e+00, -2.037500e+01], [-2.362500e+01, 1.137500e+01, -2.950000e+01], [3.775000e+01, 4.150000e+01, 4.975000e+01]]> : tensor<7x3xbf16>
    return %0 : tensor<7x3xbf16>
  }
}
