// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @expected() : () -> tensor<2x3xui16>
    %1 = stablehlo.iota dim = 0 : tensor<2x3xui16>
    %2 = stablehlo.custom_call @check.eq(%1, %0) : (tensor<2x3xui16>, tensor<2x3xui16>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @expected() -> tensor<2x3xui16> {
    %0 = stablehlo.constant dense<[[0, 0, 0], [1, 1, 1]]> : tensor<2x3xui16>
    return %0 : tensor<2x3xui16>
  }
}
