// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xi32>
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xi32>) -> tensor<2x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xi32> {
    %0 = stablehlo.constant dense<[[-2, 0, 1], [2, 0, -2]]> : tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[0xFFFFFFFE, 0.000000e+00, 1.401300e-45], [2.802600e-45, 0.000000e+00, 0xFFFFFFFE]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
