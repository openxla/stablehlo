// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @bitcast_convert_op_test_i1_to_i64() {
  %operand = stablehlo.constant dense<[0xCDEF, 0x89AB, 0x4567, 0x0123]> : tensor<4xbf16>
  %result = stablehlo.bitcast_convert %operand : (tensor<4xbf16>) -> tensor<i64>
  check.expect_eq_const %result, dense<0x0123456789ABCDEF> : tensor<i64>
  func.return
}

// -----

func.func @bitcast_convert_op_test_i64_to_f64() {
  %operand = stablehlo.constant dense<0x0123456789ABCDEF> : tensor<i64>
  %result = stablehlo.bitcast_convert %operand : (tensor<i64>) -> tensor<f64>
  check.expect_almost_eq_const %result, dense<0x0123456789ABCDEF> : tensor<f64>
  func.return
}

// -----

func.func @bitcast_convert_op_test_f64_to_i1() {
  %operand = stablehlo.constant dense<0x0123456789ABCDEF> : tensor<f64>
  %result = stablehlo.bitcast_convert %operand : (tensor<f64>) -> tensor<4xbf16>
  check.expect_eq_const %result, dense<[0xCDEF, 0x89AB, 0x4567, 0x0123]> : tensor<4xbf16>
  func.return
}
