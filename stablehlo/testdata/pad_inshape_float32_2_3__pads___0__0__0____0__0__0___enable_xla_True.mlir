// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xf32>, tensor<f32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf32>, tensor<f32>) {
    %0 = stablehlo.constant dense<[[-5.69104915E-4, -4.63130098E-4, 0.00179659645], [0.00159655546, 4.87719197E-4, 1.44607649E-4]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    return %0, %1 : tensor<2x3xf32>, tensor<f32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[-5.69104915E-4, -4.63130098E-4, 0.00179659645], [0.00159655546, 4.87719197E-4, 1.44607649E-4]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
