// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @logistic_op_test_f64() {
  %operand = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %result = stablehlo.logistic %operand : tensor<2x2xf64>
  check.expect_almost_eq_const %result,
      dense<[[0.73105857863000488, 0.88079707797788244],
             [0.95257412682243322, 0.98201379003790844]]> : tensor<2x2xf64>
  func.return
}

// -----

func.func @logistic_op_test_c128() {
  %operand = stablehlo.constant dense<(1.0, 2.0)> : tensor<complex<f64>>
  %result = stablehlo.logistic %operand : tensor<complex<f64>>
  check.expect_almost_eq_const %result,
      dense<(1.02141536417218054, 0.40343870608154248)> : tensor<complex<f64>>
  func.return
}
