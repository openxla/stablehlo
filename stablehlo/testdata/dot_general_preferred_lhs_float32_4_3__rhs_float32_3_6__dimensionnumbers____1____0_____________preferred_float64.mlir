// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xf32>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xf32>, tensor<3x6xf32>) {
    %0 = stablehlo.constant dense<[[-3.74693418, 2.34398603, 2.1041081], [-8.42414665, -0.577481806, -2.22710657], [4.59838724, -2.86711073, 0.0989064946], [0.679040133, 2.55961132, -6.06046581]]> : tensor<4x3xf32>
    %1 = stablehlo.constant dense<[[-5.99892235, 0.891323149, 1.7891463, 4.15803719, 4.00103331, 1.5682838], [3.05114746, -5.205091, 1.55728626, 1.05174971, -1.59619689, -2.16938281], [1.82395577, -0.112828031, 2.08077788, 2.36720943, -4.93377209, -2.33540392]]> : tensor<3x6xf32>
    return %0, %1 : tensor<4x3xf32>, tensor<3x6xf32>
  }
  func.func private @expected() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[33.4672165, -15.777792, 1.32462537, -8.133740e+00, -29.1142616, -15.8752012], [44.7116737, -4.25151157, -20.6054497, -40.9073067, -21.7954788, -6.757480e+00], [-36.1529427, 19.0110607, 3.96807742, 16.3389168, 22.4867916, 13.2004499], [-7.31777954, -12.0339756, -7.4095335, -8.83084774, 28.5321751, 9.66578674]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
}

