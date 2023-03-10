// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xbf16>, tensor<4x2xbf16>)
    %1 = call @expected() : () -> tensor<3x2xbf16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<3x4xbf16>, tensor<4x2xbf16>) -> tensor<3x2xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xbf16>, tensor<3x2xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xbf16>, tensor<4x2xbf16>) {
    %0 = stablehlo.constant dense<[[-5.156250e+00, -4.187500e+00, -3.562500e+00, -1.242190e+00], [6.640630e-02, 3.847660e-01, 4.082030e-01, -4.468750e+00], [-3.187500e+00, 4.570310e-01, 4.277340e-01, -2.578130e+00]]> : tensor<3x4xbf16>
    %1 = stablehlo.constant dense<[[-1.187500e+00, -1.718750e+00], [-1.898440e+00, -3.812500e+00], [-2.593750e+00, 1.195310e+00], [4.250000e+00, 1.445310e+00]]> : tensor<4x2xbf16>
    return %0, %1 : tensor<3x4xbf16>, tensor<4x2xbf16>
  }
  func.func private @expected() -> tensor<3x2xbf16> {
    %0 = stablehlo.constant dense<[[1.800000e+01, 1.875000e+01], [-2.087500e+01, -7.562500e+00], [-9.125000e+00, 5.195310e-01]]> : tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}

