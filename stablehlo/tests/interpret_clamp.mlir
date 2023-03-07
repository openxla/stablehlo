// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @clamp_op_test_si64() {
  %min = stablehlo.constant dense<[1, 5, -5]> : tensor<3xi64>
  %operand = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = stablehlo.constant dense<[3, 7, -3]> : tensor<3xi64>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  check.expect_eq_const %result, dense<[2, 5, -3]> : tensor<3xi64>
  func.return
}

// -----

func.func @clamp_op_test_si64_min_scalar() {
  %min = stablehlo.constant dense<[0, 0, -2]> : tensor<3xi64>
  %operand = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = stablehlo.constant dense<1> : tensor<i64>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<3xi64>, tensor<3xi64>, tensor<i64>) -> tensor<3xi64>
  check.expect_eq_const %result, dense<[1, 1, -1]> : tensor<3xi64>
  func.return
}

// -----

func.func @clamp_op_test_si64_max_scalar() {
  %min = stablehlo.constant dense<0> : tensor<i64>
  %operand = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = stablehlo.constant dense<[1, 1, 4]> : tensor<3xi64>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<i64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  check.expect_eq_const %result, dense<[1, 1, 0]> : tensor<3xi64>
  func.return
}

// -----

func.func @clamp_op_test_si64_min_max_both_scalar() {
  %min = stablehlo.constant dense<0> : tensor<i64>
  %operand = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = stablehlo.constant dense<1> : tensor<i64>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<i64>, tensor<3xi64>, tensor<i64>) -> tensor<3xi64>
  check.expect_eq_const %result, dense<[1, 1, 0]> : tensor<3xi64>
  func.return
}
