// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @dynamic_gather_op_test() {
  %operand = stablehlo.constant dense<[[[1, 2], [3, 4], [5, 6], [7, 8]],
                                       [[9, 10], [11, 12], [13, 14], [15, 16]],
                                       [[17, 18], [19, 20], [21, 22], [23, 24]]]> : tensor<3x4x2xi64>
  %start_indices = stablehlo.constant dense<[[[0, 0], [1, 0], [2, 1]],
                                             [[0, 1], [1, 1], [0, 9]]]> : tensor<2x3x2xi64>
  %slice_sizes = stablehlo.constant dense<[1, 2, 2]> : tensor<3xi64>
  %result = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xi64>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x2x2xi64>
  check.expect_eq_const %result, dense<[[[[1, 2], [3, 4]],
                                         [[3, 4], [5, 6]],
                                         [[13, 14], [15, 16]]],
                                        [[[9, 10], [11, 12]],
                                         [[11, 12], [13, 14]],
                                         [[17, 18], [19, 20]]]]> : tensor<2x3x2x2xi64>
  func.return
}
