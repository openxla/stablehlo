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
    %0 = stablehlo.constant dense<[[[4.937500e+00, -3.203130e-01, -3.515630e+00, -1.382810e+00], [-2.453130e+00, 1.201170e-01, -4.281250e+00, 3.015630e+00], [-2.390630e+00, 3.312500e+00, -4.968750e+00, 1.953130e+00]], [[-2.832030e-01, -5.117190e-01, -2.187500e+00, -8.544920e-02], [1.296880e+00, -3.000000e+00, -2.953130e+00, -2.687500e+00], [-4.785160e-02, -2.390630e+00, -4.781250e+00, -4.296880e-01]], [[2.609380e+00, -2.531250e+00, -5.406250e+00, -1.621090e-01], [3.109380e+00, -2.843750e+00, 6.375000e+00, 1.906250e+00], [2.718750e+00, 6.718750e-01, -1.250000e+00, 3.085940e-01]], [[-5.562500e+00, -3.093750e+00, -1.320310e+00, -1.773440e+00], [5.406250e+00, 1.062500e+00, -1.242190e+00, -1.710940e+00], [-9.492180e-01, 3.062500e+00, -4.468750e+00, -3.984380e+00]], [[3.691410e-01, 6.687500e+00, 3.625000e+00, 6.601560e-01], [2.015630e+00, -3.562500e+00, -2.234380e+00, 1.328130e+00], [-1.632810e+00, 4.101560e-01, 3.609380e+00, -2.156250e+00]], [[-2.515630e+00, 2.187500e-01, -1.726560e+00, 4.375000e+00], [7.187500e+00, -3.765630e+00, 2.859380e+00, -8.554680e-01], [-5.500000e+00, -5.906250e+00, -3.859380e+00, 1.593750e+00]], [[-1.816410e-01, 2.812500e+00, -4.937500e+00, 3.781250e+00], [2.753910e-01, -1.671880e+00, 3.468750e+00, 4.711910e-02], [4.781250e+00, 1.484380e+00, 1.304690e+00, -3.765630e+00]]]> : tensor<7x3x4xbf16>
    %1 = stablehlo.constant dense<[[2.843750e+00, 3.718750e+00, -4.531250e+00, -6.289060e-01], [1.984380e+00, -1.865230e-01, -2.765630e+00, -4.968750e+00], [2.753910e-01, 4.093750e+00, -2.406250e+00, -5.429690e-01], [9.140620e-01, 4.531250e+00, -3.218750e+00, -3.109380e+00], [-5.968750e+00, 4.406250e+00, -6.531250e+00, -2.373050e-01], [-1.000000e+00, 6.679690e-01, -4.031250e+00, -2.859380e+00], [1.046880e+00, -3.769530e-01, -2.437500e+00, -1.523440e+00]]> : tensor<7x4xbf16>
    return %0, %1 : tensor<7x3x4xbf16>, tensor<7x4xbf16>
  }
  func.func private @expected() -> tensor<7x3xbf16> {
    %0 = stablehlo.constant dense<[[2.962500e+01, 1.100000e+01, 2.675000e+01], [6.000000e+00, 2.462500e+01, 1.568750e+01], [3.453130e+00, -2.712500e+01, 6.343750e+00], [-9.312500e+00, 1.912500e+01, 3.975000e+01], [3.437500e+00, -1.343750e+01, -1.150000e+01], [-2.890630e+00, -1.875000e+01, 1.256250e+01], [5.031250e+00, -7.593750e+00, 7.000000e+00]]> : tensor<7x3xbf16>
    return %0 : tensor<7x3xbf16>
  }
}

