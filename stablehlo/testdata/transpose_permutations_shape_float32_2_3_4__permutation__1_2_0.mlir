// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3x4xf32>
    %1 = call @expected() : () -> tensor<3x4x2xf32>
    %2 = stablehlo.transpose %0, dims = [1, 2, 0] : (tensor<2x3x4xf32>) -> tensor<3x4x2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x4x2xf32>, tensor<3x4x2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3x4xf32> {
    %0 = stablehlo.constant dense<[[[1.58909702, -3.29110479, -4.5229125, -2.02355504], [-5.2291913, -2.32745957, 0.410715669, -6.9613409], [-6.73274517, 5.25102949, 1.60699403, 6.31244135]], [[0.815644443, 6.24662971, 1.3186307, -2.22678375], [-0.796898603, -4.74064922, 2.40567923, 3.60277486], [-1.24683356, 0.389498383, -2.51559758, 4.69905949]]]> : tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
  func.func private @expected() -> tensor<3x4x2xf32> {
    %0 = stablehlo.constant dense<[[[1.58909702, 0.815644443], [-3.29110479, 6.24662971], [-4.5229125, 1.3186307], [-2.02355504, -2.22678375]], [[-5.2291913, -0.796898603], [-2.32745957, -4.74064922], [0.410715669, 2.40567923], [-6.9613409, 3.60277486]], [[-6.73274517, -1.24683356], [5.25102949, 0.389498383], [1.60699403, -2.51559758], [6.31244135, 4.69905949]]]> : tensor<3x4x2xf32>
    return %0 : tensor<3x4x2xf32>
  }
}
