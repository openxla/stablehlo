// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3x4xf32>
    %1 = call @expected() : () -> tensor<2x3x4xf32>
    %2 = stablehlo.transpose %0, dims = [0, 1, 2] : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3x4xf32> {
    %0 = stablehlo.constant dense<[[[1.49130654, 2.055800e+00, 2.68664646, 0.31707117], [2.8710885, 1.03499961, -2.94407248, 0.429428458], [1.13316309, -3.29701638, -2.24880767, -3.79039526]], [[0.832629263, -0.144276917, -5.45351219, 5.26429892], [-8.70073413, 7.42637205, 0.509320259, 1.99469113], [4.247230e+00, -0.832058429, 6.70026541, -1.68749642]]]> : tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
  func.func private @expected() -> tensor<2x3x4xf32> {
    %0 = stablehlo.constant dense<[[[1.49130654, 2.055800e+00, 2.68664646, 0.31707117], [2.8710885, 1.03499961, -2.94407248, 0.429428458], [1.13316309, -3.29701638, -2.24880767, -3.79039526]], [[0.832629263, -0.144276917, -5.45351219, 5.26429892], [-8.70073413, 7.42637205, 0.509320259, 1.99469113], [4.247230e+00, -0.832058429, 6.70026541, -1.68749642]]]> : tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}
