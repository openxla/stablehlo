// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x5xi1>
    %1 = call @expected() : () -> tensor<4x5xi1>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xi1>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x5xi1>, tensor<4x5xi1>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x5xi1> {
    %0 = stablehlo.constant dense<true> : tensor<4x5xi1>
    return %0 : tensor<4x5xi1>
  }
  func.func private @expected() -> tensor<4x5xi1> {
    %0 = stablehlo.constant dense<true> : tensor<4x5xi1>
    return %0 : tensor<4x5xi1>
  }
}
