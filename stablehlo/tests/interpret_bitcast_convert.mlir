// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @bitcast_convert_op_test_i1_to_i64() {
  %operand = stablehlo.constant dense<[true, true, true, true,
                                       false, true, true, true,
                                       true, false, true, true,
                                       false, false, true, true,
                                       true, true, false, true,
                                       false, true, false, true,
                                       true, false, false, true,
                                       false, false, false, true,
                                       true, true, true, false,
                                       false, true, true, false,
                                       true, false, true, false,
                                       false, false, true, false,
                                       true, true, false, false,
                                       false, true, false, false,
                                       true, false, false, false,
                                       false, false, false, false]> : tensor<64xi1>
  %result = stablehlo.bitcast_convert %operand : (tensor<64xi1>) -> tensor<i64>
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
  %result = stablehlo.bitcast_convert %operand : (tensor<f64>) -> tensor<64xi1>
  check.expect_eq_const %result, dense<[true, true, true, true,
                                        false, true, true, true,
                                        true, false, true, true,
                                        false, false, true, true,
                                        true, true, false, true,
                                        false, true, false, true,
                                        true, false, false, true,
                                        false, false, false, true,
                                        true, true, true, false,
                                        false, true, true, false,
                                        true, false, true, false,
                                        false, false, true, false,
                                        true, true, false, false,
                                        false, true, false, false,
                                        true, false, false, false,
                                        false, false, false, false]> : tensor<64xi1>
  func.return
}

// -----

func.func @bitcast_convert_op_test_c128_to_c64() {
  %operand = stablehlo.constant dense<(0x0123456789ABCDEF, 0x0000000011111111)> : tensor<complex<f64>>
  %result = stablehlo.bitcast_convert %operand : (tensor<complex<f64>>) -> tensor<2xcomplex<f32>>
  check.expect_eq_const %result, dense<[(0x89ABCDEF, 0x01234567), (0x11111111, 0x00000000)]> : tensor<2xcomplex<f32>>
  func.return
}
