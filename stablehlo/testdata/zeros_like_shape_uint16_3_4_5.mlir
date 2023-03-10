// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4x5xui16>
    %1 = call @expected() : () -> tensor<3x4x5xui16>
    %2 = stablehlo.constant dense<0> : tensor<ui16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui16>) -> tensor<3x4x5xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x4x5xui16>, tensor<3x4x5xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4x5xui16> {
    %0 = stablehlo.constant dense<[[[0, 1, 1, 0, 0], [5, 1, 1, 7, 2], [0, 2, 0, 2, 2], [0, 0, 1, 4, 1]], [[1, 2, 1, 1, 4], [3, 1, 6, 0, 2], [0, 1, 0, 1, 1], [3, 0, 0, 2, 8]], [[1, 2, 0, 1, 1], [1, 0, 1, 1, 0], [2, 0, 4, 3, 1], [0, 0, 3, 0, 2]]]> : tensor<3x4x5xui16>
    return %0 : tensor<3x4x5xui16>
  }
  func.func private @expected() -> tensor<3x4x5xui16> {
    %0 = stablehlo.constant dense<0> : tensor<3x4x5xui16>
    return %0 : tensor<3x4x5xui16>
  }
}
