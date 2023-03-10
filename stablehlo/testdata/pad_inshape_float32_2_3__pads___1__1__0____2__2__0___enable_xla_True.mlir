// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xf32>, tensor<f32>)
    %1 = call @expected() : () -> tensor<4x7xf32>
    %2 = stablehlo.pad %0#0, %0#1, low = [1, 2], high = [1, 2], interior = [0, 0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<4x7xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x7xf32>, tensor<4x7xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf32>, tensor<f32>) {
    %0 = stablehlo.constant dense<[[7.08294741E-4, 1.64511548E-5, 1.20610239E-4], [0.00175948755, 2.10455619E-4, -4.88758553E-4]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    return %0, %1 : tensor<2x3xf32>, tensor<f32>
  }
  func.func private @expected() -> tensor<4x7xf32> {
    %0 = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 7.08294741E-4, 1.64511548E-5, 1.20610239E-4, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.00175948755, 2.10455619E-4, -4.88758553E-4, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<4x7xf32>
    return %0 : tensor<4x7xf32>
  }
}
