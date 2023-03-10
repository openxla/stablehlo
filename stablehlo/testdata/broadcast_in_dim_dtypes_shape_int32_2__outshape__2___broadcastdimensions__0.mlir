// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2xi32>
    %1 = call @expected() : () -> tensor<2xi32>
    %2 = stablehlo.custom_call @check.eq(%0, %1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2xi32> {
    %0 = stablehlo.constant dense<[-2, 0]> : tensor<2xi32>
    return %0 : tensor<2xi32>
  }
  func.func private @expected() -> tensor<2xi32> {
    %0 = stablehlo.constant dense<[-2, 0]> : tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}
