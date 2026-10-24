// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @remainder_op_test_si64() {
  %lhs = stablehlo.constant dense<[17, -17, 17, -17]> : tensor<4xi64>
  %rhs = stablehlo.constant dense<[3, 3, -3, -3]> : tensor<4xi64>
  %result = stablehlo.remainder %lhs, %rhs : tensor<4xi64>
  check.expect_eq_const %result, dense<[2, -2, 2, -2]> : tensor<4xi64>
  func.return
}

// -----

func.func @remainder_op_test_ui64() {
  %lhs = stablehlo.constant dense<[17, 18, 19, 20]> : tensor<4xui64>
  %rhs = stablehlo.constant dense<[3, 4, 5, 7]> : tensor<4xui64>
  %result = stablehlo.remainder %lhs, %rhs : tensor<4xui64>
  check.expect_eq_const %result, dense<[2, 2, 4, 6]> : tensor<4xui64>
  func.return
}

// -----

func.func @remainder_op_test_f64() {
  %lhs = stablehlo.constant dense<[17.1, -17.1, 17.1, -17.1]> : tensor<4xf64>
  %rhs = stablehlo.constant dense<[3.0, 3.0, -3.0, -3.0]> : tensor<4xf64>
  %result = stablehlo.remainder %lhs, %rhs : tensor<4xf64>
  check.expect_eq_const %result, dense<[2.1000000000000014, -2.1000000000000014, 2.1000000000000014, -2.1000000000000014]> : tensor<4xf64>
  func.return
}

// -----

// Remainder by zero is implementation-defined; this interpreter returns 0.
func.func @remainder_op_test_si64_remainder_by_zero() {
  %lhs = stablehlo.constant dense<[5, -5]> : tensor<2xi64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xi64>
  %result = stablehlo.remainder %lhs, %rhs : tensor<2xi64>
  check.expect_eq_const %result, dense<[0, 0]> : tensor<2xi64>
  func.return
}

// -----

// Remainder by zero is implementation-defined; this interpreter returns 0.
func.func @remainder_op_test_ui64_remainder_by_zero() {
  %lhs = stablehlo.constant dense<5> : tensor<ui64>
  %rhs = stablehlo.constant dense<0> : tensor<ui64>
  %result = stablehlo.remainder %lhs, %rhs : tensor<ui64>
  check.expect_eq_const %result, dense<0> : tensor<ui64>
  func.return
}

// -----

// Signed INT_MIN % -1 has a quotient that overflows the result type and is
// implementation-defined; this interpreter returns 0.
func.func @remainder_op_test_si64_signed_overflow() {
  %lhs = stablehlo.constant dense<-9223372036854775808> : tensor<i64>
  %rhs = stablehlo.constant dense<-1> : tensor<i64>
  %result = stablehlo.remainder %lhs, %rhs : tensor<i64>
  check.expect_eq_const %result, dense<0> : tensor<i64>
  func.return
}
