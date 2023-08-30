// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xf32>, tensor<2xf32>)
    %1 = call @expected() : () -> tensor<2xf32>
    %2 = stablehlo.divide %0#0, %0#1 : tensor<2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[-1.51913178, -1.08529949]> : tensor<2xf32>
    %1 = stablehlo.constant dense<[2.67503524, -1.21969211]> : tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<2xf32> {
    %0 = stablehlo.constant dense<[-0.567892253, 0.889814317]> : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
