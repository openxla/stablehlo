// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @dynamic_iota_op_test_si64_dim_0() {
  %output_shape = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %0 = stablehlo.dynamic_iota %output_shape, dim = 0 : (tensor<2xi64>) -> tensor<3x4xi64>
  check.expect_eq_const %0, dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi64>
  func.return
}

// -----

func.func @dynamic_iota_op_test_si64_dim_1() {
  %output_shape = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %0 = stablehlo.dynamic_iota %output_shape, dim = 1 : (tensor<2xi64>) -> tensor<3x4xi64>
  check.expect_eq_const %0, dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi64>
  func.return
}
