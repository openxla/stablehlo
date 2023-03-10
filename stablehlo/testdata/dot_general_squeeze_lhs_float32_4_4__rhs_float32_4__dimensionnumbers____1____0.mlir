// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xf32>, tensor<4xf32>)
    %1 = call @expected() : () -> tensor<4xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xf32>, tensor<4xf32>) {
    %0 = stablehlo.constant dense<[[2.9270215, 7.86154318, -5.63383484, 1.18890381], [1.66500914, -0.686581432, -1.0598495, 3.66114569], [-2.12638235, -5.93207598, 1.81490195, 0.333228439], [-0.129492328, 5.85269737, 1.17887712, -3.05277419]]> : tensor<4x4xf32>
    %1 = stablehlo.constant dense<[0.148809016, 4.21798277, -8.70141696, -2.01860809]> : tensor<4xf32>
    return %0, %1 : tensor<4x4xf32>, tensor<4xf32>
  }
  func.func private @expected() -> tensor<4xf32> {
    %0 = stablehlo.constant dense<[80.2178345, -0.816446066, -41.8026962, 20.5717602]> : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

