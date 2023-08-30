// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4x5xi1>
    %1 = call @expected() : () -> tensor<3x4x5xi1>
    %2 = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<3x4x5xi1>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x4x5xi1>, tensor<3x4x5xi1>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4x5xi1> {
    %0 = stablehlo.constant dense<true> : tensor<3x4x5xi1>
    return %0 : tensor<3x4x5xi1>
  }
  func.func private @expected() -> tensor<3x4x5xi1> {
    %0 = stablehlo.constant dense<false> : tensor<3x4x5xi1>
    return %0 : tensor<3x4x5xi1>
  }
}
