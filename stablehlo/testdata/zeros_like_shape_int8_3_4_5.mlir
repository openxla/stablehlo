// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4x5xi8>
    %1 = call @expected() : () -> tensor<3x4x5xi8>
    %2 = stablehlo.constant dense<0> : tensor<i8>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i8>) -> tensor<3x4x5xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x4x5xi8>, tensor<3x4x5xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4x5xi8> {
    %0 = stablehlo.constant dense<[[[0, 2, 4, 4, 0], [0, 0, 3, 0, 0], [1, 0, -6, 0, -3], [3, 5, 3, 0, -3]], [[-2, 0, 2, -1, -2], [1, 0, 0, 3, -6], [4, 1, -2, 0, 0], [2, 0, -2, 3, 1]], [[0, 4, 0, 2, -2], [-1, -1, -1, 2, 0], [4, -3, -2, 0, -1], [0, -2, -4, 0, 4]]]> : tensor<3x4x5xi8>
    return %0 : tensor<3x4x5xi8>
  }
  func.func private @expected() -> tensor<3x4x5xi8> {
    %0 = stablehlo.constant dense<0> : tensor<3x4x5xi8>
    return %0 : tensor<3x4x5xi8>
  }
}
