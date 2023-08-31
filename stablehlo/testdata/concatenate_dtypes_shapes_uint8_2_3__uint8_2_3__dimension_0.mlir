// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xui8>, tensor<2x3xui8>)
    %1 = call @expected() : () -> tensor<4x3xui8>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 0 : (tensor<2x3xui8>, tensor<2x3xui8>) -> tensor<4x3xui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x3xui8>, tensor<4x3xui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xui8>, tensor<2x3xui8>) {
    %0 = stablehlo.constant dense<[[0, 1, 3], [2, 0, 0]]> : tensor<2x3xui8>
    %1 = stablehlo.constant dense<[[1, 3, 5], [0, 1, 0]]> : tensor<2x3xui8>
    return %0, %1 : tensor<2x3xui8>, tensor<2x3xui8>
  }
  func.func private @expected() -> tensor<4x3xui8> {
    %0 = stablehlo.constant dense<[[0, 1, 3], [2, 0, 0], [1, 3, 5], [0, 1, 0]]> : tensor<4x3xui8>
    return %0 : tensor<4x3xui8>
  }
}
