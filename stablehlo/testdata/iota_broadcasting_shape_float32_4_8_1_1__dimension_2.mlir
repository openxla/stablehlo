// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @expected() : () -> tensor<4x8x1x1xf32>
    %1 = stablehlo.iota dim = 2 : tensor<4x8x1x1xf32>
    %2 = stablehlo.custom_call @check.eq(%1, %0) : (tensor<4x8x1x1xf32>, tensor<4x8x1x1xf32>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @expected() -> tensor<4x8x1x1xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<4x8x1x1xf32>
    return %0 : tensor<4x8x1x1xf32>
  }
}
