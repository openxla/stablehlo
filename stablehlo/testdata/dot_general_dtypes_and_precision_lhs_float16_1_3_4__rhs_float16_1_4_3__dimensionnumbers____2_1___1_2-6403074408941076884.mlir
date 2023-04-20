// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xf16>, tensor<1x4x3xf16>)
    %1 = call @expected() : () -> tensor<1xf16>
    %2 = stablehlo.convert %0#0 : (tensor<1x3x4xf16>) -> tensor<1x3x4xf32>
    %3 = stablehlo.convert %0#1 : (tensor<1x4x3xf16>) -> tensor<1x4x3xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>} : (tensor<1x3x4xf32>, tensor<1x4x3xf32>) -> tensor<1xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<1xf16>, tensor<1xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xf16>, tensor<1x4x3xf16>) {
    %0 = stablehlo.constant dense<[[[-1.701170e+00, 4.410160e+00, -2.992190e+00, 5.332030e+00], [3.093260e-01, -5.980460e+00, -1.592770e+00, 2.423100e-01], [-5.224610e-01, 1.324220e+00, 3.250000e+00, 2.621090e+00]]]> : tensor<1x3x4xf16>
    %1 = stablehlo.constant dense<[[[-2.468750e+00, -2.556150e-01, 6.748050e-01], [1.475590e+00, 7.524410e-01, -4.144530e+00], [-4.437500e+00, -2.604980e-01, -2.429200e-02], [-3.085940e+00, 4.519530e+00, 6.789060e+00]]]> : tensor<1x4x3xf16>
    return %0, %1 : tensor<1x3x4xf16>, tensor<1x4x3xf16>
  }
  func.func private @expected() -> tensor<1xf16> {
    %0 = stablehlo.constant dense<1.634380e+01> : tensor<1xf16>
    return %0 : tensor<1xf16>
  }
}

