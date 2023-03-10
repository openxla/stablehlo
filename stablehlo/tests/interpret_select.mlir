// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @select_op_test_si64() {
  %pred = stablehlo.constant dense<[true, false, true]> : tensor<3xi1>
  %on_true = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %on_false = stablehlo.constant dense<[3, 7, -3]> : tensor<3xi64>
  %result = stablehlo.select %pred, %on_true, %on_false : (tensor<3xi1>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  check.expect_eq_const %result, dense<[2, 7, -1]> : tensor<3xi64>
  func.return
}

// -----

func.func @select_op_test_si64_scalar() {
  %pred = stablehlo.constant dense<false> : tensor<i1>
  %on_true = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %on_false = stablehlo.constant dense<[3, 7, -3]> : tensor<3xi64>
  %result = stablehlo.select %pred, %on_true, %on_false : (tensor<i1>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  check.expect_eq_const %result, dense<[3, 7, -3]> : tensor<3xi64>
  func.return
}
