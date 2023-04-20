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
    %0 = stablehlo.constant dense<[-3.91622734, -0.242899641, -1.58674753]> : tensor<3xf32>
    %1 = stablehlo.constant dense<[[0.932529568, 1.69621098, 1.12284088, 4.28206635, 0.539385378, 2.11882901], [1.37038183, 3.67467952, -3.68408799, -0.532391131, -1.91454673, 0.2745637], [1.70064592, -0.347891033, -3.86588287, 0.385282725, -1.16977382, -2.22889447]]> : tensor<3x6xf32>
    return %0, %1 : tensor<3xf32>, tensor<3x6xf32>
  }
  func.func private @expected() -> tensor<6xf32> {
    %0 = stablehlo.constant dense<[-6.68335867, -6.9833107, 2.63174343, -17.2515736, 0.208822712, -4.82781506]> : tensor<6xf32>
    return %0 : tensor<6xf32>
  }
}

