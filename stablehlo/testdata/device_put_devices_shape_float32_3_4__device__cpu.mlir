// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4xf32>
    %1 = call @expected() : () -> tensor<3x4xf32>
    %2 = stablehlo.custom_call @check.eq(%0, %1) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4xf32> {
    %0 = stablehlo.constant dense<[[-2.3339026, -1.5990684, 2.60371065, -0.41779241], [0.172766894, 6.88342714, 6.54222536, -0.868433594], [1.14568889, -3.52940965, 0.586199641, 1.48926497]]> : tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
  func.func private @expected() -> tensor<3x4xf32> {
    %0 = stablehlo.constant dense<[[-2.3339026, -1.5990684, 2.60371065, -0.41779241], [0.172766894, 6.88342714, 6.54222536, -0.868433594], [1.14568889, -3.52940965, 0.586199641, 1.48926497]]> : tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}
