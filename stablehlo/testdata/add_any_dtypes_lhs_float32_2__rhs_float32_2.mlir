// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xf32>, tensor<2xf32>)
    %1 = call @expected() : () -> tensor<2xf32>
    %2 = stablehlo.add %0#0, %0#1 : tensor<2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[-1.90563846, -0.36767748]> : tensor<2xf32>
    %1 = stablehlo.constant dense<[-1.4110862, -4.08281088]> : tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<2xf32> {
    %0 = stablehlo.constant dense<[-3.31672478, -4.45048857]> : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
