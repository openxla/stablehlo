// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4x5xi16>
    %1 = call @expected() : () -> tensor<3x4x5xi16>
    %2 = stablehlo.constant dense<0> : tensor<i16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i16>) -> tensor<3x4x5xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x4x5xi16>, tensor<3x4x5xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4x5xi16> {
    %0 = stablehlo.constant dense<[[[0, -3, -2, -1, -5], [-1, -3, 3, 3, -2], [-5, 0, -4, 0, -1], [2, 2, -2, -2, 2]], [[-1, -3, 1, 0, 1], [-5, -3, 2, 8, 0], [-2, 0, 0, -4, 1], [-2, 0, 0, 1, 0]], [[-5, -3, 1, 0, 2], [1, 0, 4, -5, -1], [0, -3, -1, 0, 2], [0, 1, -2, -2, 1]]]> : tensor<3x4x5xi16>
    return %0 : tensor<3x4x5xi16>
  }
  func.func private @expected() -> tensor<3x4x5xi16> {
    %0 = stablehlo.constant dense<0> : tensor<3x4x5xi16>
    return %0 : tensor<3x4x5xi16>
  }
}
