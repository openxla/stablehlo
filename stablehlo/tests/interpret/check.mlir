// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @check_eq() {
  %operand = stablehlo.constant dense<[-2, 0, 2]> : tensor<3xi64>
  %result = stablehlo.constant dense<[-2, 0, 2]> : tensor<3xi64>
  check.expect_eq %operand, %result : tensor<3xi64>
  func.return
}

// -----

func.func @check_eq_const() {
  %operand = stablehlo.constant dense<[-2, 0, 2]> : tensor<3xi64>
  check.expect_eq_const %operand, dense<[-2, 0, 2]> : tensor<3xi64>
  func.return
}


// -----

func.func @check_almost_eq() {
  %operand = stablehlo.constant dense<5.0000> : tensor<f64>
  %result = stablehlo.constant dense<5.0001> : tensor<f64>
  check.expect_almost_eq %operand, %result : tensor<f64>
  func.return
}

// -----

func.func @check_almost_eq_const() {
  %operand = stablehlo.constant dense<5.0000> : tensor<f64>
  check.expect_almost_eq_const %operand, dense<5.0001> : tensor<f64>
  func.return
}

// -----

func.func @check_almost_eq_tolerance() {
  %operand = stablehlo.constant dense<5.0000> : tensor<f64>
  %result = stablehlo.constant dense<5.0001> : tensor<f64>
  check.expect_almost_eq %operand, %result, tolerance = 0.1 : tensor<f64>
  func.return
}

// -----

func.func @check_almost_eq_const_tolerance() {
  %operand = stablehlo.constant dense<5.0000> : tensor<f64>
  check.expect_almost_eq_const %operand, dense<5.1> : tensor<f64> {tolerance = 0.1 : f64}
  func.return
}

// -----

func.func @check_close() {
  %operand = stablehlo.constant dense<5.0000> : tensor<f16>
  %result = stablehlo.constant dense<5.010> : tensor<f16>
  check.expect_close %operand, %result, max_ulp_difference = 3 : tensor<f16>, tensor<f16>
  func.return
}
