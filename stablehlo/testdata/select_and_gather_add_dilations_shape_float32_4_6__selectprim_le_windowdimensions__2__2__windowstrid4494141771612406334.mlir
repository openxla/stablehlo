// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x6xf32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>):
      %6 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<f32>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      stablehlo.return %7, %8 : tensor<f32>, tensor<f32>
    }) {window_dilations = dense<[2, 3]> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<2x3xf32>, tensor<2x3xf32>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf32>, tensor<4x6xf32>) {
    %0 = stablehlo.constant dense<[[-4.38155174, 0.571365654, 0.225351438, -0.858923256, -1.91300416, -0.914961576], [-1.40737343, 3.32285762, 0.467655659, -1.69961357, -1.50950933, -0.946974575], [3.28887558, -0.189828545, 3.83203506, 1.98189867, 2.55649471, 0.645448446], [3.4715538, -3.07425237, 1.34055471, -0.340494186, 0.584194243, 3.01148939]]> : tensor<4x6xf32>
    %1 = stablehlo.constant dense<[[0.550293744, 0.519435406, -3.42726731, -5.1670866, 0.882948994, 0.923603236], [6.42240953, 0.592817843, 4.24116802, 3.6259253, 0.908526897, -5.48483276], [0.0743337423, -1.08514154, -3.78158879, -0.922211289, -0.00598636875, -4.05808067], [-3.43788695, -4.42423439, -1.64094055, -1.50363111, 3.44021869, 1.55974746]]> : tensor<4x6xf32>
    return %0, %1 : tensor<4x6xf32>, tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[-0.858923256, -0.189828545, 0.645448446], [3.4715538, -3.07425237, -0.946974575]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

