// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xf32>, tensor<f32>)
    %1 = call @expected() : () -> tensor<6x4xf32>
    %2 = stablehlo.pad %0#0, %0#1, low = [1, 0], high = [2, 1], interior = [1, 0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<6x4xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<6x4xf32>, tensor<6x4xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf32>, tensor<f32>) {
    %0 = stablehlo.constant dense<[[-4.9118686E-4, 0.00196522591, 2.68474221E-4], [3.91242065E-5, 4.20965924E-4, 3.40039289E-4]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    return %0, %1 : tensor<2x3xf32>, tensor<f32>
  }
  func.func private @expected() -> tensor<6x4xf32> {
    %0 = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-4.9118686E-4, 0.00196522591, 2.68474221E-4, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [3.91242065E-5, 4.20965924E-4, 3.40039289E-4, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<6x4xf32>
    return %0 : tensor<6x4xf32>
  }
}
