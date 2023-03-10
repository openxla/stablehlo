// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x3xf32>
    %1 = call @expected() : () -> tensor<2x1xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<[3, 2]> : tensor<2xi64>, start_indices = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x3xf32>) -> tensor<2x1xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x3xf32> {
    %0 = stablehlo.constant dense<[[-2.45863914, -1.59291494, -2.05368686], [2.93304467, -7.23922682, -2.90648675], [-2.85066676, -2.32904243, -1.90567982], [2.20618868, -2.78452539, 0.790878593], [0.956910967, -0.838458418, 0.0331095532]]> : tensor<5x3xf32>
    return %0 : tensor<5x3xf32>
  }
  func.func private @expected() -> tensor<2x1xf32> {
    %0 = stablehlo.constant dense<[[-7.23922682], [-2.32904243]]> : tensor<2x1xf32>
    return %0 : tensor<2x1xf32>
  }
}
