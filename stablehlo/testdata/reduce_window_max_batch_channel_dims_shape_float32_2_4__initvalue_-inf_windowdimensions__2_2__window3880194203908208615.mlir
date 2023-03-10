// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x4xf32>
    %1 = call @expected() : () -> tensor<1x3xf32>
    %2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<1x3xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<[[-1.54387271, -3.95520401, 0.7177096, 1.5560267], [-1.7584486, -4.21658182, -0.0695435777, -1.28604436]]> : tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  func.func private @expected() -> tensor<1x3xf32> {
    %0 = stablehlo.constant dense<[[-1.54387271, 0.7177096, 1.5560267]]> : tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

