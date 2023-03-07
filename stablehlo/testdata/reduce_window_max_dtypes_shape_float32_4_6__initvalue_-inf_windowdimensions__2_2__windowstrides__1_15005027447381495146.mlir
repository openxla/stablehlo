// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[3.29900861, -2.70639443, -2.45027924, -0.832582354, 2.40935421, -0.0693621263], [-3.18384147, -2.57581925, 1.62734878, 2.22559404, 3.71143818, 0.71347028], [-2.63500142, 2.04470801, -0.874305427, -3.90707278, -6.87103748, 3.814360e+00], [2.91417956, 0.566562414, 4.119650e+00, -1.20086825, 1.13013947, -0.670021593]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[3.29900861, 1.62734878, 2.22559404, 3.71143818, 3.71143818], [2.04470801, 2.04470801, 2.22559404, 3.71143818, 3.814360e+00], [2.91417956, 4.119650e+00, 4.119650e+00, 1.13013947, 3.814360e+00]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

