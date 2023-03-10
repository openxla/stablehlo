// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xf32>, tensor<1x4x3xf32>)
    %1 = call @expected() : () -> tensor<1xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>} : (tensor<1x3x4xf32>, tensor<1x4x3xf32>) -> tensor<1xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xf32>, tensor<1x4x3xf32>) {
    %0 = stablehlo.constant dense<[[[3.21390557, 2.41580057, -0.537137687, -1.02739561], [1.35577726, -2.48765302, 3.99296689, 3.09424305], [-2.41700459, -1.63692343, -6.27982473, 2.19841671]]]> : tensor<1x3x4xf32>
    %1 = stablehlo.constant dense<[[[-0.227901727, -0.527938426, 2.04744601], [0.46251452, -2.02832699, -5.24830675], [-1.4312098, -5.60030842, 6.4256258], [-0.754353106, -7.84103918, -1.26745594]]]> : tensor<1x4x3xf32>
    return %0, %1 : tensor<1x3x4xf32>, tensor<1x4x3xf32>
  }
  func.func private @expected() -> tensor<1xf32> {
    %0 = stablehlo.constant dense<-79.8610687> : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

