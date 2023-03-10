// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x5xui16>
    %1 = call @expected() : () -> tensor<4x5xui16>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x5xui16>, tensor<4x5xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x5xui16> {
    %0 = stablehlo.constant dense<[[5, 4, 0, 0, 3], [2, 0, 2, 0, 2], [4, 4, 3, 7, 2], [3, 2, 4, 3, 1]]> : tensor<4x5xui16>
    return %0 : tensor<4x5xui16>
  }
  func.func private @expected() -> tensor<4x5xui16> {
    %0 = stablehlo.constant dense<[[3, 2, 4, 3, 1], [4, 4, 3, 7, 2], [2, 0, 2, 0, 2], [5, 4, 0, 0, 3]]> : tensor<4x5xui16>
    return %0 : tensor<4x5xui16>
  }
}
