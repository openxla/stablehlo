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
    %0 = stablehlo.constant dense<[[1.51487315, 1.84060729, 1.371032], [-2.20898318, -5.76905251, -2.57896972], [-1.58193374, -2.16250372, -4.66199541], [0.347719163, 8.04571342, -9.58430767]]> : tensor<4x3xf32>
    %1 = stablehlo.constant dense<[-0.860122799, -1.25689316, -3.9581697]> : tensor<3xf32>
    return %0, %1 : tensor<4x3xf32>, tensor<3xf32>
  }
  func.func private @expected() -> tensor<4xf32> {
    %0 = stablehlo.constant dense<[-9.04320049, 19.3590794, 22.531662, 27.5246334]> : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

