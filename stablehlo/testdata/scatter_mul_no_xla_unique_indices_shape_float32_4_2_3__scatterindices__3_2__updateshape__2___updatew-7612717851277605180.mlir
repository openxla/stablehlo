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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf32>, tensor<2xi32>, tensor<2xf32>) -> tensor<4x2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[[[-1.96583712, 2.72805405, 6.03738785], [0.876314282, 3.04074049, 3.28707695]], [[-0.782448232, 2.99008107, -3.60445976], [-2.17568398, 4.26613474, -2.22315931]], [[3.78778887, -2.07535934, 0.770484328], [3.07007432, -1.00540853, 2.158041]], [[2.83093143, -2.77950644, -1.88555098], [1.98335791, -1.62500179, 1.58836579]]]> : tensor<4x2x3xf32>
    %1 = stablehlo.constant dense<[-0.186029136, 2.68265462]> : tensor<2xf32>
    return %0, %1 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<4x2x3xf32> {
    %0 = stablehlo.constant dense<[[[-1.96583712, 2.72805405, 6.03738785], [0.876314282, 3.04074049, 3.28707695]], [[-0.782448232, 2.99008107, -3.60445976], [-2.17568398, 4.26613474, -2.22315931]], [[3.78778887, -2.07535934, 0.770484328], [3.07007432, -1.00540853, 2.158041]], [[2.83093143, -2.77950644, 0.350767434], [1.98335791, -1.62500179, 4.26103687]]]> : tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}

