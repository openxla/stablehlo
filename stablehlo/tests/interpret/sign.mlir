// RUN: stablehlo-translate --interpret -split-input-file %s

// CHECK-LABEL: Evaluated results of function: sign_op_test_si64
func.func @sign_op_test_si64() {
  %operand = stablehlo.constant dense<[-1, 0, 1]> : tensor<3xi64>
  %result = stablehlo.sign %operand : tensor<3xi64>
  check.expect_eq_const %result, dense<[-1, 0, 1]> : tensor<3xi64>
  func.return
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_f64
func.func @sign_op_test_f64() {
  // +NaN, -1.0, -0.0, +0.0, 1.0
  %operand = stablehlo.constant dense<[0x7FFFFFFFFFFFFFFF, -1.0, -0.0, 0.0, 1.0]> : tensor<5xf64>
  %result = stablehlo.sign %operand : tensor<5xf64>
  check.expect_almost_eq_const %result, dense<[0x7FFFFFFFFFFFFFFF, -1.0, -0.0, 0.0, 1.0]> : tensor<5xf64>
  func.return
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_c128
func.func @sign_op_test_c128() {
  // (+NaN, +0.0), (+0.0, +NaN), (0.0, 0.0), (0.0, 1.0)
  %operand = stablehlo.constant dense<[(0x7FF0000000000001, 0x0000000000000000),
                                       (0x0000000000000000, 0x7FF0000000000001),
                                       (0.0, 0.0),
                                       (0.0, 1.0)]> : tensor<4xcomplex<f64>>
  %result = stablehlo.sign %operand : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(0x7FF0000000000001, 0x7FF0000000000001),
                                               (0x7FF0000000000001, 0x7FF0000000000001),
                                               (0.0, 0.0),
                                               (0.0, 1.0)]> : tensor<4xcomplex<f64>>
  func.return
}
