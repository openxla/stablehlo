// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xf32>, tensor<f32>)
    %1 = call @expected() : () -> tensor<2x1xf32>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, -1], high = [0, -1], interior = [0, 0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x1xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf32>, tensor<f32>) {
    %0 = stablehlo.constant dense<[[7.86395511E-4, -0.00205520424, -8.4031286E-4], [7.32857734E-4, -1.49261235E-4, -0.00100950524]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    return %0, %1 : tensor<2x3xf32>, tensor<f32>
  }
  func.func private @expected() -> tensor<2x1xf32> {
    %0 = stablehlo.constant dense<[[-0.00205520424], [-1.49261235E-4]]> : tensor<2x1xf32>
    return %0 : tensor<2x1xf32>
  }
}
