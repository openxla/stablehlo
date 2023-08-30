// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<ui32>, tensor<ui32>)
    %1 = call @expected() : () -> tensor<i1>
    %2 = stablehlo.compare  GT, %0#0, %0#1,  UNSIGNED : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<ui32>, tensor<ui32>) {
    %0 = stablehlo.constant dense<3> : tensor<ui32>
    %1 = stablehlo.constant dense<3> : tensor<ui32>
    return %0, %1 : tensor<ui32>, tensor<ui32>
  }
  func.func private @expected() -> tensor<i1> {
    %0 = stablehlo.constant dense<false> : tensor<i1>
    return %0 : tensor<i1>
  }
}
