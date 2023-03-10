// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xf32>, tensor<4x2xf32>)
    %1 = call @expected() : () -> tensor<3x2xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xf32>, tensor<4x2xf32>) {
    %0 = stablehlo.constant dense<[[-0.648610889, -4.839990e-01, 3.39964437, 0.349830806], [-4.12569952, -6.90287971, -0.153646722, 5.38082075], [-2.10003686, -0.173380762, 2.26172876, 1.9670006]]> : tensor<3x4xf32>
    %1 = stablehlo.constant dense<[[1.9392488, -1.40549958], [-3.80043983, 3.44176579], [-3.12474394, 0.0999774113], [-2.64203429, -2.605490e+00]]> : tensor<4x2xf32>
    return %0, %1 : tensor<3x4xf32>, tensor<4x2xf32>
  }
  func.func private @expected() -> tensor<3x2xf32> {
    %0 = stablehlo.constant dense<[[-10.9656916, -1.32578194], [4.49701452, -31.994463], [-15.6777773, -2.54401326]]> : tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}

