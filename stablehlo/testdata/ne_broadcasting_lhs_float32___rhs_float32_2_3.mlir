// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<f32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x3xi1>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %3 = stablehlo.compare  NE, %2, %0#1,  FLOAT : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<f32>, tensor<2x3xf32>) {
    %0 = stablehlo.constant dense<[[3.12061596, 2.09694242, -2.50136518], [-4.69077301, 0.0742209628, 1.50642514]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<0.229106382> : tensor<f32>
    return %1, %0 : tensor<f32>, tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<2x3xi1> {
    %0 = stablehlo.constant dense<true> : tensor<2x3xi1>
    return %0 : tensor<2x3xi1>
  }
}
