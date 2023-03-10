// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[3, 2]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3xf32>, tensor<2xf32>)
    %2 = call @expected() : () -> tensor<4x2x3xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf32>, tensor<2xi32>, tensor<2xf32>) -> tensor<4x2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[[[-0.68528676, 2.13478732, 1.37833309], [3.13193703, -4.37889242, -2.85170913]], [[-3.41413498, 1.63311756, -1.00111628], [-4.102370e+00, 0.695868492, -4.23566866]], [[2.84877586, -1.76168764, -3.29387808], [-2.07421494, -0.981891751, 1.62912965]], [[-6.5848918, 1.29409444, -3.81396604], [-0.264611423, 0.329450458, 0.890426636]]]> : tensor<4x2x3xf32>
    %1 = stablehlo.constant dense<[2.3444736, -5.52479839]> : tensor<2xf32>
    return %0, %1 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<4x2x3xf32> {
    %0 = stablehlo.constant dense<[[[-0.68528676, 2.13478732, 1.37833309], [3.13193703, -4.37889242, -2.85170913]], [[-3.41413498, 1.63311756, -1.00111628], [-4.102370e+00, 0.695868492, -4.23566866]], [[2.84877586, -1.76168764, -3.29387808], [-2.07421494, -0.981891751, 1.62912965]], [[-6.5848918, 1.29409444, 2.3444736], [-0.264611423, 0.329450458, 0.890426636]]]> : tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}

