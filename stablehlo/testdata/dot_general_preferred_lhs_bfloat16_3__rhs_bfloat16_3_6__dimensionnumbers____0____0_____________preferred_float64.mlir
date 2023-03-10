// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xbf16>, tensor<3x6xbf16>)
    %1 = call @expected() : () -> tensor<6xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xbf16>, tensor<3x6xbf16>) -> tensor<6xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xbf16>, tensor<3x6xbf16>) {
    %0 = stablehlo.constant dense<[-1.093750e+00, -2.453130e+00, 4.160160e-01]> : tensor<3xbf16>
    %1 = stablehlo.constant dense<[[-9.609370e-01, 4.296880e-01, 9.687500e-01, 2.203130e+00, -2.890630e-01, 1.191410e-01], [-3.964840e-01, -1.476560e+00, -7.226560e-01, -1.460940e+00, 6.805420e-03, 3.015630e+00], [4.804690e-01, 1.085940e+00, 1.804690e+00, -1.226560e+00, -3.046880e+00, 1.765630e+00]]> : tensor<3x6xbf16>
    return %0, %1 : tensor<3xbf16>, tensor<3x6xbf16>
  }
  func.func private @expected() -> tensor<6xf32> {
    %0 = stablehlo.constant dense<[2.22353363, 3.60398865, 1.463974, 0.663925171, -0.968080043, -6.79348755]> : tensor<6xf32>
    return %0 : tensor<6xf32>
  }
}

