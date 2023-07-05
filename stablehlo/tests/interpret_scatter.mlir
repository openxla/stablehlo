// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @scatter_op_test() {
  %inputs = stablehlo.constant dense<[[[1, 2], [3, 4], [5, 6], [7, 8]],
                                      [[9, 10], [11, 12], [13, 14], [15, 16]],
                                      [[17, 18], [19, 20], [21, 22], [23, 24]]]> : tensor<3x4x2xi64>
  %scatter_indices = stablehlo.constant dense<[[[0, 2], [1, 0], [2, 1]],
                                               [[0, 1], [1, 0], [0, 9]]]> : tensor<2x3x2xi64>
  %updates = stablehlo.constant dense<1> : tensor<2x3x2x2xi64>
  %result = "stablehlo.scatter"(%inputs, %scatter_indices, %updates) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
      stablehlo.return %0 : tensor<i64>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<3x4x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2x2xi64>) -> tensor<3x4x2xi64>
  check.expect_eq_const %result, dense<[[[1, 2], [5, 6], [7, 8], [7, 8]],
                                        [[10, 11], [12, 13], [14, 15], [16, 17]],
                                        [[18, 19], [20, 21], [21, 22], [23, 24]]]> : tensor<3x4x2xi64>
  func.return
}
