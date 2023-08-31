// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xui8>, tensor<2xui8>)
    %1 = call @expected() : () -> tensor<2xui8>
    %2 = stablehlo.add %0#0, %0#1 : tensor<2xui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xui8>, tensor<2xui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xui8>, tensor<2xui8>) {
    %0 = stablehlo.constant dense<[0, 1]> : tensor<2xui8>
    %1 = stablehlo.constant dense<4> : tensor<2xui8>
    return %0, %1 : tensor<2xui8>, tensor<2xui8>
  }
  func.func private @expected() -> tensor<2xui8> {
    %0 = stablehlo.constant dense<[4, 5]> : tensor<2xui8>
    return %0 : tensor<2xui8>
  }
}
