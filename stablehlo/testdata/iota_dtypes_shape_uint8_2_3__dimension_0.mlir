// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @expected() : () -> tensor<2x3xui8>
    %1 = stablehlo.iota dim = 0 : tensor<2x3xui8>
    %2 = stablehlo.custom_call @check.eq(%1, %0) : (tensor<2x3xui8>, tensor<2x3xui8>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @expected() -> tensor<2x3xui8> {
    %0 = stablehlo.constant dense<[[0, 0, 0], [1, 1, 1]]> : tensor<2x3xui8>
    return %0 : tensor<2x3xui8>
  }
}
