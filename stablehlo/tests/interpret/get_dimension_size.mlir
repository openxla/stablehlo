// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @get_dimension_size_op_test_si64() {
  %operand = stablehlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
  %result = stablehlo.get_dimension_size %operand, dim = 1 : (tensor<2x3xi64>) -> tensor<i32>
  check.expect_eq_const %result, dense<3> : tensor<i32>
  func.return
}
