// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xf32>, tensor<3xf32>)
    %1 = call @expected() : () -> tensor<f32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xf32>, tensor<3xf32>) {
    %0 = stablehlo.constant dense<[1.03572977, 1.25679076, -0.538284242]> : tensor<3xf32>
    %1 = stablehlo.constant dense<[1.03247929, -0.158634588, -3.32975769]> : tensor<3xf32>
    return %0, %1 : tensor<3xf32>, tensor<3xf32>
  }
  func.func private @expected() -> tensor<f32> {
    %0 = stablehlo.constant dense<2.66235518> : tensor<f32>
    return %0 : tensor<f32>
  }
}

