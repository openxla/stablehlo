// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xf32>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<6xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xf32>, tensor<3x6xf32>) -> tensor<6xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xf32>, tensor<3x6xf32>) {
    %0 = stablehlo.constant dense<[4.54189587, -0.56804496, 2.11517882]> : tensor<3xf32>
    %1 = stablehlo.constant dense<[[4.05453825, -1.44036746, 4.79499197, 2.67171788, 0.616522252, -4.48888302], [4.79667568, -4.19831371, -1.44450212, 2.85810256, -0.367232859, 1.95318091], [-1.62081826, -5.7897296, 1.62217569, -0.311252445, 1.74422383, -0.186609924]]> : tensor<3x6xf32>
    return %0, %1 : tensor<3xf32>, tensor<3x6xf32>
  }
  func.func private @expected() -> tensor<6xf32> {
    %0 = stablehlo.constant dense<[12.2622423, -16.4034805, 26.0300884, 9.85277938, 6.698130e+00, -21.8922462]> : tensor<6xf32>
    return %0 : tensor<6xf32>
  }
}

