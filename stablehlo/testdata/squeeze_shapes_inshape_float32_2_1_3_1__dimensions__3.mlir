// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x1x3x1xf32>
    %1 = call @expected() : () -> tensor<2x1x3xf32>
    %2 = stablehlo.reshape %0 : (tensor<2x1x3x1xf32>) -> tensor<2x1x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x1x3xf32>, tensor<2x1x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x1x3x1xf32> {
    %0 = stablehlo.constant dense<[[[[0.292375416], [2.41566968], [-2.352355]]], [[[-1.49155188], [-4.60985327], [4.23529673]]]]> : tensor<2x1x3x1xf32>
    return %0 : tensor<2x1x3x1xf32>
  }
  func.func private @expected() -> tensor<2x1x3xf32> {
    %0 = stablehlo.constant dense<[[[0.292375416, 2.41566968, -2.352355]], [[-1.49155188, -4.60985327, 4.23529673]]]> : tensor<2x1x3xf32>
    return %0 : tensor<2x1x3xf32>
  }
}
