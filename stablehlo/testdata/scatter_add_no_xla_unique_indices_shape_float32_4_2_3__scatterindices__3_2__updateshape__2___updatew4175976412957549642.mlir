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
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf32>, tensor<2xi32>, tensor<2xf32>) -> tensor<4x2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[[[-3.26196623, 4.7657795, -2.50036764], [-2.95160031, 0.463332355, -2.94465494]], [[-0.804442405, -1.60302448, -1.89281881], [-1.48239458, -3.82481146, 3.14968228]], [[1.11707628, 0.756418585, -1.30863607], [4.27384043, 4.33618879, -1.94776881]], [[-0.803710877, -0.229005456, -0.18369332], [2.45710802, 1.55194128, 2.48913169]]]> : tensor<4x2x3xf32>
    %1 = stablehlo.constant dense<[2.60337114, -1.87256396]> : tensor<2xf32>
    return %0, %1 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<4x2x3xf32> {
    %0 = stablehlo.constant dense<[[[-3.26196623, 4.7657795, -2.50036764], [-2.95160031, 0.463332355, -2.94465494]], [[-0.804442405, -1.60302448, -1.89281881], [-1.48239458, -3.82481146, 3.14968228]], [[1.11707628, 0.756418585, -1.30863607], [4.27384043, 4.33618879, -1.94776881]], [[-0.803710877, -0.229005456, 2.41967773], [2.45710802, 1.55194128, 0.616567731]]]> : tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}

