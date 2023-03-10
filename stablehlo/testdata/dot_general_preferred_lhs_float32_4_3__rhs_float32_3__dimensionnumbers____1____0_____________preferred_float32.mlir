// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xf32>, tensor<3xf32>)
    %1 = call @expected() : () -> tensor<4xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xf32>, tensor<3xf32>) -> tensor<4xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xf32>, tensor<3xf32>) {
    %0 = stablehlo.constant dense<[[-3.57498646, 2.26188898, 2.9657414], [-1.29635787, -4.13106441, -3.49819064], [4.14597797, -0.373203397, -2.34024167], [5.10803795, -3.48006749, -3.4904089]]> : tensor<4x3xf32>
    %1 = stablehlo.constant dense<[5.14886427, -1.20347404, -0.895334362]> : tensor<3xf32>
    return %0, %1 : tensor<4x3xf32>, tensor<3xf32>
  }
  func.func private @expected() -> tensor<4xf32> {
    %0 = stablehlo.constant dense<[-23.7845745, 1.42890823, 23.8915176, 33.6138496]> : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

