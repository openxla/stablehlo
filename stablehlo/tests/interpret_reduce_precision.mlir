// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @reduce_precision_op_test_f64() {
  %operand = stablehlo.constant dense<[0x7FF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0.0, 65505.0, 65520.0]> : tensor<6xf64>
  %output = stablehlo.reduce_precision %operand, format = e5m10 : tensor<6xf64>
  check.expect_almost_eq_const %output, dense<[0x7FF0000000000000, 0x7FFFFFFFFFFFFFFF, 0.0, 0.0, 65504.0, 0x7FF0000000000000]> : tensor<6xf64>
  func.return
}

// -----

func.func @reduce_precision_op_test_f64_zero_mantissa_bits() {
  %operand = stablehlo.constant dense<0x7FFFFFFFFFFFFFFF> : tensor<f64>
  %output = stablehlo.reduce_precision %operand, format = e5m0 : tensor<f64>
  check.expect_almost_eq_const %output, dense<0x7FF0000000000000> : tensor<f64>
  func.return
}
