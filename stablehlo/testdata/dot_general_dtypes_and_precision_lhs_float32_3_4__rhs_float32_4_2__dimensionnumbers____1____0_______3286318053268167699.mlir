// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xf32>, tensor<4x2xf32>)
    %1 = call @expected() : () -> tensor<3x2xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xf32>, tensor<4x2xf32>) {
    %0 = stablehlo.constant dense<[[-2.98286057, 1.45473039, 4.37298536, 4.06851482], [-3.93359685, 5.34374475, -7.59047747, -1.32499611], [-4.2501359, 4.22664261, 3.23046494, 7.55214739]]> : tensor<3x4xf32>
    %1 = stablehlo.constant dense<[[-2.42577767, 1.8425138], [-2.80951762, -6.45209217], [0.841208279, -3.3215766], [0.892308592, -3.71017742]]> : tensor<4x2xf32>
    return %0, %1 : tensor<3x4xf32>, tensor<4x2xf32>
  }
  func.func private @expected() -> tensor<3x2xf32> {
    %0 = stablehlo.constant dense<[[10.4576283, -44.5021362], [-13.0387917, -11.5977163], [7.89139795, -73.8516693]]> : tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}
