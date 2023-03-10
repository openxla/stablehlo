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
    %0 = stablehlo.constant dense<[[5.81311798, 2.08485532, 0.151162371], [-1.21007407, -1.59476554, 0.846119463], [-0.83784312, -0.416278511, 1.24929118], [3.46354723, 2.21915126, 3.81866336]]> : tensor<4x3xf32>
    %1 = stablehlo.constant dense<[[-2.10215521, -1.803730e+00, -7.83739519, 4.36787844, 1.4788357, 3.10357666], [-4.46420813, 0.879630148, -2.18081808, -1.95115197, -3.56435633, -0.671983778], [-2.76886797, -0.212248296, 2.77085519, -1.21441388, -3.28464937, -4.60568237]]> : tensor<3x6xf32>
    return %0, %1 : tensor<4x3xf32>, tensor<3x6xf32>
  }
  func.func private @expected() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[-21.9458523, -8.6834774, -49.6875458, 21.1395512, 0.668963671, 15.9442625], [7.32033587, 0.600255728, 15.3061972, -3.20136595, 1.11560607, -6.58085871], [0.160507768, 0.87991172, 10.9359407, -4.36453056, -3.85875082, -8.07441616], [-27.7610416, -5.10577631, -21.4037914, 6.1610136, -15.3307991, -8.329400e+00]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
}

