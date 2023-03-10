// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<f32>, tensor<2x3xf32>, tensor<f32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %3 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %4 = stablehlo.clamp %2, %0#1, %3 : tensor<2x3xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<f32>, tensor<2x3xf32>, tensor<f32>) {
    %0 = stablehlo.constant dense<[[-0.84212619, -2.22140336, -6.88380194], [-2.94575262, -0.124904193, -3.84827161]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<-2.58011985> : tensor<f32>
    %2 = stablehlo.constant dense<-1.20042491> : tensor<f32>
    return %1, %0, %2 : tensor<f32>, tensor<2x3xf32>, tensor<f32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[-1.20042491, -2.22140336, -2.58011985], [-2.58011985, -1.20042491, -2.58011985]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
