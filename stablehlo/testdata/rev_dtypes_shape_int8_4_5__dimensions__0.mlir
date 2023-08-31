// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x5xi8>
    %1 = call @expected() : () -> tensor<4x5xi8>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xi8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x5xi8>, tensor<4x5xi8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x5xi8> {
    %0 = stablehlo.constant dense<[[-7, 1, 2, 2, 1], [-4, -3, -1, 2, 1], [0, 1, 0, -2, 2], [-1, 0, -1, 1, 0]]> : tensor<4x5xi8>
    return %0 : tensor<4x5xi8>
  }
  func.func private @expected() -> tensor<4x5xi8> {
    %0 = stablehlo.constant dense<[[-1, 0, -1, 1, 0], [0, 1, 0, -2, 2], [-4, -3, -1, 2, 1], [-7, 1, 2, 2, 1]]> : tensor<4x5xi8>
    return %0 : tensor<4x5xi8>
  }
}
