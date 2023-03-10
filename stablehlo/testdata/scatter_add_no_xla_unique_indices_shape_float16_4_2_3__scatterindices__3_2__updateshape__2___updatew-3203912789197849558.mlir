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
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf16>, tensor<2xi32>, tensor<2xf16>) -> tensor<4x2x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16>, tensor<2xf16>) {
    %0 = stablehlo.constant dense<[[[-1.559570e+00, -1.922850e+00, 1.263670e+00], [3.460940e+00, 1.130860e+00, -3.427730e+00]], [[-2.630860e+00, 1.326170e+00, -3.572270e+00], [-2.878420e-01, 9.501950e-01, -2.882810e+00]], [[6.921880e+00, -4.687500e+00, 6.953130e-01], [2.728520e+00, 2.007810e+00, 3.322270e+00]], [[3.648440e+00, 8.696280e-01, -6.889640e-01], [5.473630e-01, -4.109380e+00, -5.830080e-01]]]> : tensor<4x2x3xf16>
    %1 = stablehlo.constant dense<[-1.977540e+00, 5.000000e+00]> : tensor<2xf16>
    return %0, %1 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<4x2x3xf16> {
    %0 = stablehlo.constant dense<[[[-1.559570e+00, -1.922850e+00, 1.263670e+00], [3.460940e+00, 1.130860e+00, -3.427730e+00]], [[-2.630860e+00, 1.326170e+00, -3.572270e+00], [-2.878420e-01, 9.501950e-01, -2.882810e+00]], [[6.921880e+00, -4.687500e+00, 6.953130e-01], [2.728520e+00, 2.007810e+00, 3.322270e+00]], [[3.648440e+00, 8.696280e-01, -2.666020e+00], [5.473630e-01, -4.109380e+00, 4.417970e+00]]]> : tensor<4x2x3xf16>
    return %0 : tensor<4x2x3xf16>
  }
}

