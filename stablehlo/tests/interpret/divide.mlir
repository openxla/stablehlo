// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @divide_op_test_si64() {
  %lhs = stablehlo.constant dense<[17, -17, 17, -17]> : tensor<4xi64>
  %rhs = stablehlo.constant dense<[3, 3, -3, -3]> : tensor<4xi64>
  %result = stablehlo.divide %lhs, %rhs : tensor<4xi64>
  check.expect_eq_const %result, dense<[5, -5, -5, 5]> : tensor<4xi64>
  func.return
}

// -----

func.func @divide_op_test_ui64() {
  %lhs = stablehlo.constant dense<[17, 18, 19, 20]> : tensor<4xui64>
  %rhs = stablehlo.constant dense<[3, 4, 5, 7]> : tensor<4xui64>
  %result = stablehlo.divide %lhs, %rhs : tensor<4xui64>
  check.expect_eq_const %result, dense<[5, 4, 3, 2]> : tensor<4xui64>
  func.return
}

// -----

func.func @divide_op_test_f64() {
  %lhs = stablehlo.constant dense<[17.1, -17.1, 17.1, -17.1]> : tensor<4xf64>
  %rhs = stablehlo.constant dense<[3.0, 3.0, -3.0, -3.0]> : tensor<4xf64>
  %result = stablehlo.divide %lhs, %rhs : tensor<4xf64>
  check.expect_almost_eq_const %result, dense<[5.700000e+00, -5.700000e+00, -5.700000e+00, 5.700000e+00]> : tensor<4xf64>
  func.return
}

// -----

func.func @divide_op_test_c128() {
  %lhs = stablehlo.constant dense<[(1.5, 2.5), (7.5, 5.5)]> : tensor<2xcomplex<f64>>
  %rhs = stablehlo.constant dense<[(2.5, 1.5), (5.5, 7.5)]> : tensor<2xcomplex<f64>>
  %result = stablehlo.divide %lhs, %rhs : tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(0.88235294117647056, 0.4705882352941177), (0.95375722543352603, -0.30057803468208094)]> : tensor<2xcomplex<f64>>
  func.return
}

// -----

// Division by zero is implementation-defined; this interpreter returns 0.
func.func @divide_op_test_si64_divide_by_zero() {
  %lhs = stablehlo.constant dense<[5, -5]> : tensor<2xi64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xi64>
  %result = stablehlo.divide %lhs, %rhs : tensor<2xi64>
  check.expect_eq_const %result, dense<[0, 0]> : tensor<2xi64>
  func.return
}

// -----

// Division by zero is implementation-defined; this interpreter returns 0.
func.func @divide_op_test_ui64_divide_by_zero() {
  %lhs = stablehlo.constant dense<5> : tensor<ui64>
  %rhs = stablehlo.constant dense<0> : tensor<ui64>
  %result = stablehlo.divide %lhs, %rhs : tensor<ui64>
  check.expect_eq_const %result, dense<0> : tensor<ui64>
  func.return
}

// -----

// Signed INT_MIN / -1 overflows the result type and is implementation-
// defined; this interpreter returns 0.
func.func @divide_op_test_si64_signed_overflow() {
  %lhs = stablehlo.constant dense<-9223372036854775808> : tensor<i64>
  %rhs = stablehlo.constant dense<-1> : tensor<i64>
  %result = stablehlo.divide %lhs, %rhs : tensor<i64>
  check.expect_eq_const %result, dense<0> : tensor<i64>
  func.return
}
