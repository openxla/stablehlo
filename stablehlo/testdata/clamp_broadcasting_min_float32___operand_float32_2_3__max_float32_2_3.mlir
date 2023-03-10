// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %3 = stablehlo.clamp %2, %0#1, %0#2 : tensor<2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = stablehlo.constant dense<[[4.8751874, 2.23340082, 5.61325788], [4.75495768, -1.64026928, 0.49328804]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<[[-3.03970265, 2.25319886, -3.69029379], [-2.45202589, 1.10378492, 1.25856316]]> : tensor<2x3xf32>
    %2 = stablehlo.constant dense<-2.21625209> : tensor<f32>
    return %2, %0, %1 : tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[-3.03970265, 2.23340082, -3.69029379], [-2.45202589, -1.64026928, 0.49328804]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
