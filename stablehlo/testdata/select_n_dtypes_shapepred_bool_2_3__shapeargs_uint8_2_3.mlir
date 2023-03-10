// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<2x3xi1>, tensor<2x3xui8>, tensor<2x3xui8>)
    %1 = call @expected() : () -> tensor<2x3xui8>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<2x3xi1>, tensor<2x3xui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xui8>, tensor<2x3xui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi1>, tensor<2x3xui8>, tensor<2x3xui8>) {
    %0 = stablehlo.constant dense<true> : tensor<2x3xi1>
    %1 = stablehlo.constant dense<[[4, 0, 0], [0, 6, 5]]> : tensor<2x3xui8>
    %2 = stablehlo.constant dense<[[1, 1, 5], [1, 1, 3]]> : tensor<2x3xui8>
    return %0, %1, %2 : tensor<2x3xi1>, tensor<2x3xui8>, tensor<2x3xui8>
  }
  func.func private @expected() -> tensor<2x3xui8> {
    %0 = stablehlo.constant dense<[[1, 1, 5], [1, 1, 3]]> : tensor<2x3xui8>
    return %0 : tensor<2x3xui8>
  }
}
