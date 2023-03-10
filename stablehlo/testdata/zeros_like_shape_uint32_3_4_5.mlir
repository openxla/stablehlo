// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4x5xui32>
    %1 = call @expected() : () -> tensor<3x4x5xui32>
    %2 = stablehlo.constant dense<0> : tensor<ui32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<3x4x5xui32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x4x5xui32>, tensor<3x4x5xui32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4x5xui32> {
    %0 = stablehlo.constant dense<[[[0, 2, 2, 0, 1], [1, 3, 1, 5, 4], [1, 2, 1, 4, 1], [0, 4, 1, 2, 4]], [[0, 1, 1, 1, 0], [3, 0, 3, 3, 9], [0, 2, 2, 4, 2], [3, 2, 6, 3, 0]], [[2, 0, 0, 2, 0], [1, 0, 1, 3, 3], [2, 2, 0, 3, 2], [3, 1, 0, 3, 6]]]> : tensor<3x4x5xui32>
    return %0 : tensor<3x4x5xui32>
  }
  func.func private @expected() -> tensor<3x4x5xui32> {
    %0 = stablehlo.constant dense<0> : tensor<3x4x5xui32>
    return %0 : tensor<3x4x5xui32>
  }
}
