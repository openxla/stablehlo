// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @gather() {
  %operand = stablehlo.constant dense<[[[1, 2], [3, 4], [5, 6], [7, 8]],
                                       [[9, 10], [11, 12], [13, 14], [15, 16]],
                                       [[17, 18], [19, 20], [21, 22], [23, 24]]]> : tensor<3x4x2xi64>
  %start_indices = stablehlo.constant dense<[[[0, 0], [1, 0], [2, 1]],
                                             [[0, 1], [1, 1], [0, 2]]]> : tensor<2x3x2xi64>
  %result = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 3],
      collapsed_slice_dims = [2],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = dense<[2, 2, 1]> : tensor<3xi64>,
    indices_are_sorted = false
  } : (tensor<3x4x2xi64>, tensor<2x3x2xi64>) -> tensor<2x2x3x2xi64>
  check.expect_eq_const %result, dense<[[[[1, 3], [3, 5], [13, 15]],
                                         [[9, 11], [11, 13], [21, 23]]],
                                        [[[9, 11], [11, 13], [9, 11]],
                                         [[17, 19], [19, 21], [17, 19]]]]> : tensor<2x2x3x2xi64>
  func.return
}
