// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @convert_op_test_i1_to_i1() {
  %operand = stablehlo.constant dense<[true, false]> : tensor<2xi1>
  %result = stablehlo.convert %operand : (tensor<2xi1>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[true, false]> : tensor<2xi1>
  func.return
}

// -----

func.func @convert_op_test_i1_to_si64() {
  %operand = stablehlo.constant dense<[true, false]> : tensor<2xi1>
  %result = stablehlo.convert %operand : (tensor<2xi1>) -> tensor<2xi64>
  check.expect_eq_const %result, dense<[1, 0]> : tensor<2xi64>
  func.return
}

// -----

func.func @convert_op_test_i1_to_ui64() {
  %operand = stablehlo.constant dense<[true, false]> : tensor<2xi1>
  %result = stablehlo.convert %operand : (tensor<2xi1>) -> tensor<2xui64>
  check.expect_eq_const %result, dense<[1, 0]> : tensor<2xui64>
  func.return
}

// -----

func.func @convert_op_test_i1_to_f64() {
  %operand = stablehlo.constant dense<[true, false]> : tensor<2xi1>
  %result = stablehlo.convert %operand : (tensor<2xi1>) -> tensor<2xf64>
  check.expect_almost_eq_const %result, dense<[1.0, 0.0]> : tensor<2xf64>
  func.return
}

// -----

func.func @convert_op_test_i1_to_c128() {
  %operand = stablehlo.constant dense<[true, false]> : tensor<2xi1>
  %result = stablehlo.convert %operand : (tensor<2xi1>) -> tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(1.0, 0.0), (0.0, 0.0)]> : tensor<2xcomplex<f64>>
  func.return
}

// -----

func.func @convert_op_test_si64_to_i1() {
  %operand = stablehlo.constant dense<[-1, 0, 1]> : tensor<3xi64>
  %result = stablehlo.convert %operand : (tensor<3xi64>) -> tensor<3xi1>
  check.expect_eq_const %result, dense<[true, false, true]> : tensor<3xi1>
  func.return
}

// -----

func.func @convert_op_test_si64_to_si64() {
  %operand = stablehlo.constant dense<[-1, 0, 1]> : tensor<3xi64>
  %result = stablehlo.convert %operand : (tensor<3xi64>) -> tensor<3xi64>
  check.expect_eq_const %result, dense<[-1, 0, 1]> : tensor<3xi64>
  func.return
}

// -----

func.func @convert_op_test_si64_to_ui64() {
  %operand = stablehlo.constant dense<[0, 1]> : tensor<2xi4>
  %result = stablehlo.convert %operand : (tensor<2xi4>) -> tensor<2xui64>
  check.expect_eq_const %result, dense<[0, 1]> : tensor<2xui64>
  func.return
}

// -----

func.func @convert_op_test_si64_to_f64() {
  %operand = stablehlo.constant dense<[-1, 0, 1]> : tensor<3xi64>
  %result = stablehlo.convert %operand : (tensor<3xi64>) -> tensor<3xf64>
  check.expect_almost_eq_const %result, dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
  func.return
}

// -----

func.func @convert_op_test_si64_to_c128() {
  %operand = stablehlo.constant dense<[-1, 0, 1]> : tensor<3xi4>
  %result = stablehlo.convert %operand : (tensor<3xi4>) -> tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]> : tensor<3xcomplex<f64>>
  func.return
}

// -----

func.func @convert_op_test_ui64_to_i1() {
  %operand = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %result = stablehlo.convert %operand : (tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[false, true]> : tensor<2xi1>
  func.return
}

// -----

func.func @convert_op_test_ui64_to_si64() {
  %operand = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %result = stablehlo.convert %operand : (tensor<2xui64>) -> tensor<2xi64>
  check.expect_eq_const %result, dense<[0, 1]> : tensor<2xi64>
  func.return
}

// -----

func.func @convert_op_test_ui64_to_ui64() {
  %operand = stablehlo.constant dense<[0, 1]> : tensor<2xui4>
  %result = stablehlo.convert %operand : (tensor<2xui4>) -> tensor<2xui64>
  check.expect_eq_const %result, dense<[0, 1]> : tensor<2xui64>
  func.return
}

// -----

func.func @convert_op_test_ui64_to_f64() {
  %operand = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %result = stablehlo.convert %operand : (tensor<2xui64>) -> tensor<2xf64>
  check.expect_almost_eq_const %result, dense<[0.0, 1.0]> : tensor<2xf64>
  func.return
}

// -----

func.func @convert_op_test_ui64_to_c128() {
  %operand = stablehlo.constant dense<[0, 1]> : tensor<2xui4>
  %result = stablehlo.convert %operand : (tensor<2xui4>) -> tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(0.0, 0.0), (1.0, 0.0)]> : tensor<2xcomplex<f64>>
  func.return
}

// -----

func.func @convert_op_test_f64_to_i1() {
  %operand = stablehlo.constant dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
  %result = stablehlo.convert %operand : (tensor<3xf64>) -> tensor<3xi1>
  check.expect_eq_const %result, dense<[true, false, true]> : tensor<3xi1>
  func.return
}

// -----

func.func @convert_op_test_f64_to_si64() {
  %operand = stablehlo.constant dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
  %result = stablehlo.convert %operand : (tensor<3xf64>) -> tensor<3xi64>
  check.expect_eq_const %result, dense<[-1, 0, 1]> : tensor<3xi64>
  func.return
}

// -----

func.func @convert_op_test_f64_to_ui64() {
  %operand = stablehlo.constant dense<[0.0, 1.0]> : tensor<2xf64>
  %result = stablehlo.convert %operand : (tensor<2xf64>) -> tensor<2xui64>
  check.expect_eq_const %result, dense<[0, 1]> : tensor<2xui64>
  func.return
}

// -----

func.func @convert_op_test_f64_to_f64() {
  %operand = stablehlo.constant dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
  %result = stablehlo.convert %operand : (tensor<3xf64>) -> tensor<3xf64>
  check.expect_almost_eq_const %result, dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
  func.return
}

// -----

func.func @convert_op_test_f64_to_c128() {
  %operand = stablehlo.constant dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
  %result = stablehlo.convert %operand : (tensor<3xf64>) -> tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]> : tensor<3xcomplex<f64>>
  func.return
}

// -----

func.func @convert_op_test_c128_to_i1() {
  %operand = stablehlo.constant dense<[(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]> : tensor<3xcomplex<f64>>
  %result = stablehlo.convert %operand : (tensor<3xcomplex<f64>>) -> tensor<3xi1>
  check.expect_eq_const %result, dense<[true, false, true]> : tensor<3xi1>
  func.return
}

// -----

func.func @convert_op_test_c128_to_si64() {
  %operand = stablehlo.constant dense<[(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]> : tensor<3xcomplex<f64>>
  %result = stablehlo.convert %operand : (tensor<3xcomplex<f64>>) -> tensor<3xi64>
  check.expect_eq_const %result, dense<[-1, 0, 1]> : tensor<3xi64>
  func.return
}

// -----

func.func @convert_op_test_c128_to_ui64() {
  %operand = stablehlo.constant dense<[(0.0, 1.0), (1.0, 0.0)]> : tensor<2xcomplex<f64>>
  %result = stablehlo.convert %operand : (tensor<2xcomplex<f64>>) -> tensor<2xui64>
  check.expect_eq_const %result, dense<[0, 1]> : tensor<2xui64>
  func.return
}

// -----

func.func @convert_op_test_c128_to_f64() {
  %operand = stablehlo.constant dense<[(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]> : tensor<3xcomplex<f64>>
  %result = stablehlo.convert %operand : (tensor<3xcomplex<f64>>) -> tensor<3xf64>
  check.expect_almost_eq_const %result, dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
  func.return
}

// -----

func.func @convert_op_test_c128_to_c128() {
  %operand = stablehlo.constant dense<[(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]> : tensor<3xcomplex<f64>>
  %result = stablehlo.convert %operand : (tensor<3xcomplex<f64>>) -> tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]> : tensor<3xcomplex<f64>>
  func.return
}

// -----

func.func @convert_op_test_f32_to_f8e8m0fnu() {
  %operand = stablehlo.constant dense<[0x80000001, -1.0, -0.0, 0.0, 1.0, 2.0, 0xFF800000, 0xFFC00000, 0x7F800000]> : tensor<9xf32>
  %result = stablehlo.convert %operand : (tensor<9xf32>) -> tensor<9xf8E8M0FNU>
  check.expect_almost_eq_const %result, dense<[0xFF, 0xFF, 0xFF, 5.877470e-39, 1.0, 2.0, 0xFF, 0xFF, 0xFF]> : tensor<9xf8E8M0FNU>
  func.return
}

// -----

func.func @convert_op_test_bf16_to_f8e8m0fnu() {
  %operand = stablehlo.constant dense<[-1.0, -0.0, 0.0, 1.0, 0xFF80, 0xFFC0]> : tensor<6xbf16>
  %result = stablehlo.convert %operand : (tensor<6xbf16>) -> tensor<6xf8E8M0FNU>
  check.expect_almost_eq_const %result, dense<[0xFF, 0xFF, 5.877470e-39, 1.0, 0xFF, 0xFF]> : tensor<6xf8E8M0FNU>
  func.return
}

// -----

func.func @convert_op_test_si32_to_f8e8m0fnu() {
  %operand = stablehlo.constant dense<[-2, -1, 0, 1, 2]> : tensor<5xi32>
  %result = stablehlo.convert %operand : (tensor<5xi32>) -> tensor<5xf8E8M0FNU>
  check.expect_almost_eq_const %result, dense<[0xFF, 0xFF, 5.877470e-39, 1.0, 2.0]> : tensor<5xf8E8M0FNU>
  func.return
}

// -----

func.func @convert_op_test_ui8_to_f8e8m0fnu() {
  %operand = stablehlo.constant dense<[0, 1, 2, 128]> : tensor<4xui8>
  %result = stablehlo.convert %operand : (tensor<4xui8>) -> tensor<4xf8E8M0FNU>
  check.expect_almost_eq_const %result, dense<[5.877470e-39, 1.0, 2.0, 128.0]> : tensor<4xf8E8M0FNU>
  func.return
}
