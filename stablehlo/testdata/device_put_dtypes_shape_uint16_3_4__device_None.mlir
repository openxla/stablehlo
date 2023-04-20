// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4xui16>
    %1 = call @expected() : () -> tensor<3x4xui16>
    %2 = stablehlo.custom_call @check.eq(%0, %1) : (tensor<3x4xui16>, tensor<3x4xui16>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4xui16> {
    %0 = stablehlo.constant dense<[[2, 4, 4, 2], [5, 0, 2, 2], [7, 2, 0, 2]]> : tensor<3x4xui16>
    return %0 : tensor<3x4xui16>
  }
  func.func private @expected() -> tensor<3x4xui16> {
    %0 = stablehlo.constant dense<[[2, 4, 4, 2], [5, 0, 2, 2], [7, 2, 0, 2]]> : tensor<3x4xui16>
    return %0 : tensor<3x4xui16>
  }
}
