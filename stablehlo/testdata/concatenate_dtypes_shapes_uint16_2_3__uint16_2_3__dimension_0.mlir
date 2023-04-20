// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xui16>, tensor<2x3xui16>)
    %1 = call @expected() : () -> tensor<4x3xui16>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 0 : (tensor<2x3xui16>, tensor<2x3xui16>) -> tensor<4x3xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x3xui16>, tensor<4x3xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xui16>, tensor<2x3xui16>) {
    %0 = stablehlo.constant dense<[[1, 0, 0], [0, 1, 0]]> : tensor<2x3xui16>
    %1 = stablehlo.constant dense<[[1, 3, 0], [2, 0, 1]]> : tensor<2x3xui16>
    return %0, %1 : tensor<2x3xui16>, tensor<2x3xui16>
  }
  func.func private @expected() -> tensor<4x3xui16> {
    %0 = stablehlo.constant dense<[[1, 0, 0], [0, 1, 0], [1, 3, 0], [2, 0, 1]]> : tensor<4x3xui16>
    return %0 : tensor<4x3xui16>
  }
}
