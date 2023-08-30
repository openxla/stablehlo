// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<2x3xi1>, tensor<2x3xui16>, tensor<2x3xui16>)
    %1 = call @expected() : () -> tensor<2x3xui16>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<2x3xi1>, tensor<2x3xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xui16>, tensor<2x3xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi1>, tensor<2x3xui16>, tensor<2x3xui16>) {
    %0 = stablehlo.constant dense<true> : tensor<2x3xi1>
    %1 = stablehlo.constant dense<[[3, 4, 1], [2, 8, 3]]> : tensor<2x3xui16>
    %2 = stablehlo.constant dense<[[1, 0, 3], [3, 4, 3]]> : tensor<2x3xui16>
    return %0, %1, %2 : tensor<2x3xi1>, tensor<2x3xui16>, tensor<2x3xui16>
  }
  func.func private @expected() -> tensor<2x3xui16> {
    %0 = stablehlo.constant dense<[[1, 0, 3], [3, 4, 3]]> : tensor<2x3xui16>
    return %0 : tensor<2x3xui16>
  }
}
