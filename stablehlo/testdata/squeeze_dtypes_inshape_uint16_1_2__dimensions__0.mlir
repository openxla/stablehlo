// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<1x2xui16>
    %1 = call @expected() : () -> tensor<2xui16>
    %2 = stablehlo.reshape %0 : (tensor<1x2xui16>) -> tensor<2xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xui16>, tensor<2xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<1x2xui16> {
    %0 = stablehlo.constant dense<[[3, 1]]> : tensor<1x2xui16>
    return %0 : tensor<1x2xui16>
  }
  func.func private @expected() -> tensor<2xui16> {
    %0 = stablehlo.constant dense<[3, 1]> : tensor<2xui16>
    return %0 : tensor<2xui16>
  }
}
