// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[3, 2]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3xf16>, tensor<2xf16>)
    %2 = call @expected() : () -> tensor<4x2x3xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf16>, tensor<2xi32>, tensor<2xf16>) -> tensor<4x2x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16>, tensor<2xf16>) {
    %0 = stablehlo.constant dense<[[[-5.558590e+00, 2.068360e+00, 7.617180e+00], [-9.536130e-01, 4.070310e+00, 2.244140e+00]], [[-6.703130e+00, 6.142580e-01, 9.404290e-01], [9.819330e-01, 1.742190e+00, 3.011720e+00]], [[3.453130e+00, 3.132810e+00, -8.710930e-01], [2.500000e+00, 2.453130e+00, 2.841800e+00]], [[6.860350e-01, 2.166020e+00, -2.076170e+00], [-6.093750e+00, -7.011710e-01, -1.606450e+00]]]> : tensor<4x2x3xf16>
    %1 = stablehlo.constant dense<[-2.583980e+00, -2.187500e-01]> : tensor<2xf16>
    return %0, %1 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<4x2x3xf16> {
    %0 = stablehlo.constant dense<[[[-5.558590e+00, 2.068360e+00, 7.617180e+00], [-9.536130e-01, 4.070310e+00, 2.244140e+00]], [[-6.703130e+00, 6.142580e-01, 9.404290e-01], [9.819330e-01, 1.742190e+00, 3.011720e+00]], [[3.453130e+00, 3.132810e+00, -8.710930e-01], [2.500000e+00, 2.453130e+00, 2.841800e+00]], [[6.860350e-01, 2.166020e+00, -2.076170e+00], [-6.093750e+00, -7.011710e-01, -2.187500e-01]]]> : tensor<4x2x3xf16>
    return %0 : tensor<4x2x3xf16>
  }
}

