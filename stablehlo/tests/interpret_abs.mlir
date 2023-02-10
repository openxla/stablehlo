// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @abs_op_test_si64() {
  %operand = stablehlo.constant dense<[-2, 0, 2]> : tensor<3xi64>
  %result = stablehlo.abs %operand : tensor<3xi64>
  check.expect_eq_const %result, dense<[2, 0, 2]> : tensor<3xi64>
  func.return
}

// -----

func.func @abs_op_test_f8_e4m3_fnuz() {
  %operand = stablehlo.constant dense<[24.0, -24.0, -0.0, 0x80]> : tensor<4xf8E4M3FNUZ>
  %result = stablehlo.abs %operand : tensor<4xf8E4M3FNUZ>
  check.expect_almost_eq_const %result, dense<[24.0, 24.0, 0.0, 0x80]> : tensor<4xf8E4M3FNUZ>
  func.return
}

// -----

func.func @abs_op_test_f8_e5m2_fnuz() {
  %operand = stablehlo.constant dense<[24.0, -24.0, -0.0, 0x80]> : tensor<4xf8E5M2FNUZ>
  %result = stablehlo.abs %operand : tensor<4xf8E5M2FNUZ>
  check.expect_almost_eq_const %result, dense<[24.0, 24.0, 0.0, 0x80]> : tensor<4xf8E5M2FNUZ>
  func.return
}

// -----

func.func @abs_op_test_f64() {
  %operand = stablehlo.constant dense<[23.1, -23.1, -0.0]> : tensor<3xf64>
  %result = stablehlo.abs %operand : tensor<3xf64>
  check.expect_almost_eq_const %result, dense<[2.310000e+01, 2.310000e+01, 0.000000e+00]> : tensor<3xf64>
  func.return
}

// -----

func.func @abs_op_test_c64() {
  %operand = stablehlo.constant dense<(3.0, 4.0)> : tensor<complex<f64>>
  %result = "stablehlo.abs"(%operand) : (tensor<complex<f64>>) -> tensor<f64>
  check.expect_almost_eq_const %result, dense<5.000000e+00> : tensor<f64>
  func.return
}
