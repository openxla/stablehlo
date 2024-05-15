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

// -----

func.func @scatter_op_with_batching_dim_test() {
  %inputs = stablehlo.constant dense<[
      [
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        [[9, 10], [11, 12], [13, 14], [15, 16]],
        [[17, 18], [19, 20], [21, 22], [23, 24]]
      ],
      [
        [[25, 26], [27, 28], [29, 30], [31, 32]],
        [[33, 34], [35, 36], [37, 38], [39, 40]],
        [[41, 42], [43, 44], [45, 46], [47, 48]]
      ]
  ]> : tensor<2x3x4x2xi64>
  %scatter_indices = stablehlo.constant dense<[
      [
        [[0, 0], [1, 0], [2, 1]],
        [[0, 1], [1, 1], [0, 9]]
      ],
      [
        [[0, 0], [2, 1], [2, 2]],
        [[1, 2], [0, 1], [1, 0]]
      ]
  ]> : tensor<2x2x3x2xi64>
  %updates = stablehlo.constant dense<1> : tensor<2x2x3x2x2xi64>
  %result = "stablehlo.scatter"(%inputs, %scatter_indices, %updates) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
      stablehlo.return %0 : tensor<i64>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [3, 4],
      inserted_window_dims = [1],
      input_batching_dims = [0],
      scatter_indices_batching_dims = [1],
      scatter_dims_to_operand_dims = [2, 1],
      index_vector_dim = 3>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
  check.expect_eq_const %result, dense<[
      [
        [[3, 4], [6, 7], [6, 7], [7, 8]],
        [[9, 10],[11, 12], [15, 16], [17, 18]],
        [[17, 18], [19, 20], [22, 23], [24, 25]]
      ],
      [
        [[25, 26], [28, 29], [30, 31], [31, 32]],
        [[35, 36], [38, 39], [38, 39], [39, 40]],
        [[41, 42], [44, 45], [46, 47], [47, 48]]
      ]
  ]> : tensor<2x3x4x2xi64>
  func.return
}
