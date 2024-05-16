// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @gather_op_test() {
  %operand = stablehlo.constant dense<[[[1, 2], [3, 4], [5, 6], [7, 8]],
                                       [[9, 10], [11, 12], [13, 14], [15, 16]],
                                       [[17, 18], [19, 20], [21, 22], [23, 24]]]> : tensor<3x4x2xi64>
  %start_indices = stablehlo.constant dense<[[[0, 0], [1, 0], [2, 1]],
                                             [[0, 1], [1, 1], [0, 9]]]> : tensor<2x3x2xi64>
  %result = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xi64>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi64>
  check.expect_eq_const %result, dense<[[[[1, 2], [3, 4]],
                                         [[3, 4], [5, 6]],
                                         [[13, 14], [15, 16]]],
                                        [[[9, 10], [11, 12]],
                                         [[11, 12], [13, 14]],
                                         [[17, 18], [19, 20]]]]> : tensor<2x3x2x2xi64>
  func.return
}

// -----

func.func @gather_op_with_batching_dim_test() {
  %operand = stablehlo.constant dense<[
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
  %start_indices = stablehlo.constant dense<[
      [
        [[0, 0], [1, 0], [2, 1]],
        [[0, 1], [1, 1], [0, 9]]
      ],
      [
        [[0, 0], [2, 1], [2, 2]],
        [[1, 2], [0, 1], [1, 0]]
      ]
  ]> : tensor<2x2x3x2xi64>
  %result = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [3, 4],
      collapsed_slice_dims = [1],
      operand_batching_dims = [0],
      start_indices_batching_dims = [1],
      start_index_map = [2, 1],
      index_vector_dim = 3>,
    slice_sizes = array<i64: 1, 1, 2, 2>,
    indices_are_sorted = false
  } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi64>
  check.expect_eq_const %result, dense<[
      [
        [
          [[1, 2], [3, 4]],
          [[3, 4], [5, 6]],
          [[13, 14], [15, 16]]
        ],
        [
          [[33, 34], [35, 36]],
          [[35, 36], [37, 38]],
          [[41, 42], [43, 44]]
        ]
      ],
      [
        [
          [[1, 2], [3, 4]],
          [[13, 14], [15, 16]],
          [[21, 22], [23, 24]]
        ],
        [
          [[43, 44], [45, 46]],
          [[33, 34], [35, 36]],
          [[27, 28], [29, 30]]
        ]
      ]
  ]> : tensor<2x2x3x2x2xi64>
  func.return
}
