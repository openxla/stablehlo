// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {window_dimensions = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<4x6xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[6.20513773, -0.62325412, -1.09830678, 2.6595552, 3.78790522, 4.59650707], [-1.67918193, -3.11266875, -0.639529526, -2.23243618, -5.15142965, 1.71529663], [-0.83703351, 1.62865424, -2.44362092, 1.65968478, -3.98622727, 0.387630939], [1.044550e+00, -3.14918447, -3.069242, 0.211742982, -1.56455946, 5.28841734]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[6.20513773, -0.62325412, -1.09830678, 2.6595552, 3.78790522, 4.59650707], [-1.67918193, -3.11266875, -0.639529526, -2.23243618, -5.15142965, 1.71529663], [-0.83703351, 1.62865424, -2.44362092, 1.65968478, -3.98622727, 0.387630939], [1.044550e+00, -3.14918447, -3.069242, 0.211742982, -1.56455946, 5.28841734]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
}

