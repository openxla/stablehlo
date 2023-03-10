// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xf32>, tensor<4x2xf32>)
    %1 = call @expected() : () -> tensor<3x2xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xf32>, tensor<4x2xf32>) {
    %0 = stablehlo.constant dense<[[0.30003655, 3.43623924, -5.00657272, -5.54167175], [-0.0709157884, 5.499670e+00, 3.123830e-01, 0.863251984], [2.37204719, 0.134123445, -0.936426699, 1.57510769]]> : tensor<3x4xf32>
    %1 = stablehlo.constant dense<[[1.22435701, 1.81742287], [-2.47757602, 0.965225696], [-3.48883629, 0.23426415], [-2.27880955, 0.550582886]]> : tensor<4x2xf32>
    return %0, %1 : tensor<3x4xf32>, tensor<4x2xf32>
  }
  func.func private @expected() -> tensor<3x2xf32> {
    %0 = stablehlo.constant dense<[[21.9493351, -0.361970454], [-16.7697163, 5.72801065], [2.24960017, 5.08832836]]> : tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}
