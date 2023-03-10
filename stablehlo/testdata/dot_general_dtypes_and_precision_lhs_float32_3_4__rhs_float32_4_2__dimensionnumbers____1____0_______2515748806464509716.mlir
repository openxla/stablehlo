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
    %0 = stablehlo.constant dense<[[-0.988849937, -1.00252235, 1.64629126, 2.86447835], [4.66840506, -1.25282645, 1.20502043, -1.91219449], [0.263320625, 1.15289724, 0.175304011, -0.978201329]]> : tensor<3x4xf32>
    %1 = stablehlo.constant dense<[[-0.117480382, -3.2312851], [0.462347686, -3.56523347], [4.92760229, 1.94133246], [0.446654767, 0.681284487]]> : tensor<4x2xf32>
    return %0, %1 : tensor<3x4xf32>, tensor<4x2xf32>
  }
  func.func private @expected() -> tensor<3x2xf32> {
    %0 = stablehlo.constant dense<[[9.04435825, 11.9170055], [3.9560833, -9.58173274], [0.929014563, -5.28732157]]> : tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}

