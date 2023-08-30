// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xi32>
    %1 = call @expected() : () -> tensor<3x2xi32>
    %2 = stablehlo.reshape %0 : (tensor<2x3xi32>) -> tensor<3x2xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xi32> {
    %0 = stablehlo.constant dense<[[6, 1, -1], [-1, -2, 1]]> : tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
  func.func private @expected() -> tensor<3x2xi32> {
    %0 = stablehlo.constant dense<[[6, 1], [-1, -1], [-2, 1]]> : tensor<3x2xi32>
    return %0 : tensor<3x2xi32>
  }
}
