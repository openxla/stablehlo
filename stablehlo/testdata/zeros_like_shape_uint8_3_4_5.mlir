// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4x5xui8>
    %1 = call @expected() : () -> tensor<3x4x5xui8>
    %2 = stablehlo.constant dense<0> : tensor<ui8>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui8>) -> tensor<3x4x5xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x4x5xui8>, tensor<3x4x5xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4x5xui8> {
    %0 = stablehlo.constant dense<[[[3, 2, 2, 3, 0], [1, 3, 0, 1, 3], [1, 1, 1, 0, 1], [1, 3, 0, 3, 0]], [[6, 5, 3, 0, 4], [5, 0, 1, 1, 5], [5, 3, 4, 3, 0], [0, 5, 0, 0, 1]], [[4, 0, 1, 6, 3], [2, 0, 1, 2, 2], [0, 1, 0, 1, 2], [0, 5, 4, 3, 4]]]> : tensor<3x4x5xui8>
    return %0 : tensor<3x4x5xui8>
  }
  func.func private @expected() -> tensor<3x4x5xui8> {
    %0 = stablehlo.constant dense<0> : tensor<3x4x5xui8>
    return %0 : tensor<3x4x5xui8>
  }
}
