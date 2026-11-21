// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @compare_op_test_si64_signed() {
  %lhs = stablehlo.constant dense<-2> : tensor<i64>
  %rhs = stablehlo.constant dense<-2> : tensor<i64>
  %result = stablehlo.compare EQ, %lhs, %rhs, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
  check.expect_eq_const %result, dense<true> : tensor<i1>
  func.return
}

// -----

func.func @compare_op_test_si64() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare EQ, %lhs, %rhs : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[true, false, true, false, true]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare_op_test_si64() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare NE, %lhs, %rhs : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[false, true, false, true, false]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare_op_test_si64() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare GE, %lhs, %rhs : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[true, true, true, true, true]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare_op_test_si64() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare GT, %lhs, %rhs : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[false, true, false, true, false]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare_op_test_si64() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare LE, %lhs, %rhs : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[true, false, true, false, true]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare_op_test_si64() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare LT, %lhs, %rhs : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[false, false, false, false, false]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare_op_test_ui64_unsigned() {
  %lhs = stablehlo.constant dense<0> : tensor<ui64>
  %rhs = stablehlo.constant dense<0> : tensor<ui64>
  %result = stablehlo.compare EQ, %lhs, %rhs, UNSIGNED : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
  check.expect_eq_const %result, dense<true> : tensor<i1>
  func.return
}

// -----

func.func @compare_op_test_ui64() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare EQ, %lhs, %rhs : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[true, false]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare_op_test_ui64() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare NE, %lhs, %rhs : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[false, true]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare_op_test_ui64() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare GE, %lhs, %rhs : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[true, true]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare_op_test_ui64() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare GT, %lhs, %rhs : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[false, true]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare_op_test_ui64() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare LE, %lhs, %rhs : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[true, false]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare_op_test_ui64() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare LT, %lhs, %rhs : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[false, false]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare_op_test_i1_default() {
  %lhs = stablehlo.constant dense<true> : tensor<i1>
  %rhs = stablehlo.constant dense<true> : tensor<i1>
  %result = stablehlo.compare EQ, %lhs, %rhs : (tensor<i1>, tensor<i1>) -> tensor<i1>
  check.expect_eq_const %result, dense<true> : tensor<i1>
  func.return
}

// -----

func.func @compare_op_test_i1() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare EQ, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[true, false, false, true]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare_op_test_i1() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare NE, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, true, true, false]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare_op_test_i1() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare GE, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[true, true, false, true]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare_op_test_i1() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare GT, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, true, false, false]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare_op_test_i1() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare LE, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[true, false, true, true]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare_op_test_i1() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare LT, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, false, true, false]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare_op_test_f64_default() {
  // -NaN
  // -NaN
  %lhs = stablehlo.constant dense<0xFFF0000000000001> : tensor<f64>
  %rhs = stablehlo.constant dense<0xFFF0000000000001> : tensor<f64>
  %result = stablehlo.compare EQ, %lhs, %rhs : (tensor<f64>, tensor<f64>) -> tensor<i1>
  check.expect_eq_const %result, dense<false> : tensor<i1>
  func.return
}

// -----

func.func @compare_op_test_f64() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare EQ, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, false, true, false, true, false, true, true, true, false, true, true, false, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare_op_test_f64() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare NE, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[true, true, false, true, false, true, false, false, false, true, false, false, true, true]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare_op_test_f64() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare GE, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, false, true, false, true, false, true, true, true, false, true, true, false, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare_op_test_f64() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare GT, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare_op_test_f64() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare LE, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, false, true, true, true, true, true, true, true, true, true, true, false, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare_op_test_f64() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare LT, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, false, false, true, false, true, false, false, false, true, false, false, false, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare_op_test_f64_weakorder_eq() {
  // -NaN, -NaN, -Inf, -0.0, 1.0, +Inf, +Inf, -NaN
  // -NaN, +NaN, +Inf, +0.0, 2.0, +Inf, -NaN, +Inf
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0x8000000000000000, 1.0, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000001]> : tensor<8xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0x7FF0000000000000, 0x0000000000000000, 2.0, 0x7FF0000000000000, 0xFFF0000000000001, 0x7FF0000000000000]> : tensor<8xf64>
  %result = stablehlo.compare EQ, %lhs, %rhs, WEAKORDER : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xi1>
  check.expect_eq_const %result, dense<[true, true, false, true, false, true, false, false]> : tensor<8xi1>
  func.return
}

// -----

func.func @compare_op_test_f64_weakorder_ne() {
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0x8000000000000000, 1.0, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000001]> : tensor<8xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0x7FF0000000000000, 0x0000000000000000, 2.0, 0x7FF0000000000000, 0xFFF0000000000001, 0x7FF0000000000000]> : tensor<8xf64>
  %result = stablehlo.compare NE, %lhs, %rhs, WEAKORDER : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xi1>
  check.expect_eq_const %result, dense<[false, false, true, false, true, false, true, true]> : tensor<8xi1>
  func.return
}

// -----

func.func @compare_op_test_f64_weakorder_ge() {
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0x8000000000000000, 1.0, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000001]> : tensor<8xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0x7FF0000000000000, 0x0000000000000000, 2.0, 0x7FF0000000000000, 0xFFF0000000000001, 0x7FF0000000000000]> : tensor<8xf64>
  %result = stablehlo.compare GE, %lhs, %rhs, WEAKORDER : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xi1>
  check.expect_eq_const %result, dense<[true, true, false, true, false, true, false, true]> : tensor<8xi1>
  func.return
}

// -----

func.func @compare_op_test_f64_weakorder_gt() {
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0x8000000000000000, 1.0, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000001]> : tensor<8xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0x7FF0000000000000, 0x0000000000000000, 2.0, 0x7FF0000000000000, 0xFFF0000000000001, 0x7FF0000000000000]> : tensor<8xf64>
  %result = stablehlo.compare GT, %lhs, %rhs, WEAKORDER : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xi1>
  check.expect_eq_const %result, dense<[false, false, false, false, false, false, false, true]> : tensor<8xi1>
  func.return
}

// -----

func.func @compare_op_test_f64_weakorder_le() {
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0x8000000000000000, 1.0, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000001]> : tensor<8xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0x7FF0000000000000, 0x0000000000000000, 2.0, 0x7FF0000000000000, 0xFFF0000000000001, 0x7FF0000000000000]> : tensor<8xf64>
  %result = stablehlo.compare LE, %lhs, %rhs, WEAKORDER : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xi1>
  check.expect_eq_const %result, dense<[true, true, true, true, true, true, true, false]> : tensor<8xi1>
  func.return
}

// -----

func.func @compare_op_test_f64_weakorder_lt() {
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0x8000000000000000, 1.0, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000001]> : tensor<8xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0x7FF0000000000000, 0x0000000000000000, 2.0, 0x7FF0000000000000, 0xFFF0000000000001, 0x7FF0000000000000]> : tensor<8xf64>
  %result = stablehlo.compare LT, %lhs, %rhs, WEAKORDER : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xi1>
  check.expect_eq_const %result, dense<[false, false, true, false, true, false, true, false]> : tensor<8xi1>
  func.return
}

// -----

func.func @compare_op_test_f64_totalorder() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %eq = stablehlo.compare EQ, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  %ne = stablehlo.compare NE, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  %ge = stablehlo.compare GE, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  %gt = stablehlo.compare GT, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  %le = stablehlo.compare LE, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  %lt = stablehlo.compare LT, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %eq, dense<[true, false, true, false, true, false, true, false, true, false, true, true, true, false]> : tensor<14xi1>
  check.expect_eq_const %ne, dense<[false, true, false, true, false, true, false, true, false, true, false, false, false, true]> : tensor<14xi1>
  check.expect_eq_const %ge, dense<[true, false, true, false, true, false, true, false, true, false, true, true, true, false]> : tensor<14xi1>
  check.expect_eq_const %gt, dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<14xi1>
  check.expect_eq_const %le, dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<14xi1>
  check.expect_eq_const %lt, dense<[false, true, false, true, false, true, false, true, false, true, false, false, false, true]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare_op_test_f16_totalorder_lt() {
  %lhs = stablehlo.constant dense<[0xFC01, 0xFC00, 0x8000, 0x0000, 0x7C00, 0x7C01]> : tensor<6xf16>
  %rhs = stablehlo.constant dense<[0xFC00, 0x8000, 0x0000, 0x7C00, 0x7C01, 0x7C01]> : tensor<6xf16>
  %result = stablehlo.compare LT, %lhs, %rhs, TOTALORDER : (tensor<6xf16>, tensor<6xf16>) -> tensor<6xi1>
  check.expect_eq_const %result, dense<[true, true, true, true, true, false]> : tensor<6xi1>
  func.return
}

// -----

func.func @compare_op_test_bf16_totalorder_lt() {
  %lhs = stablehlo.constant dense<[0xFF81, 0xFF80, 0x8000, 0x0000, 0x7F80, 0x7F81]> : tensor<6xbf16>
  %rhs = stablehlo.constant dense<[0xFF80, 0x8000, 0x0000, 0x7F80, 0x7F81, 0x7F81]> : tensor<6xbf16>
  %result = stablehlo.compare LT, %lhs, %rhs, TOTALORDER : (tensor<6xbf16>, tensor<6xbf16>) -> tensor<6xi1>
  check.expect_eq_const %result, dense<[true, true, true, true, true, false]> : tensor<6xi1>
  func.return
}

// -----

func.func @compare_op_test_f32_totalorder_lt() {
  %lhs = stablehlo.constant dense<[0xFF800001, 0xFF800000, 0x80000000, 0x00000000, 0x7F800000, 0x7F800001]> : tensor<6xf32>
  %rhs = stablehlo.constant dense<[0xFF800000, 0x80000000, 0x00000000, 0x7F800000, 0x7F800001, 0x7F800001]> : tensor<6xf32>
  %result = stablehlo.compare LT, %lhs, %rhs, TOTALORDER : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
  check.expect_eq_const %result, dense<[true, true, true, true, true, false]> : tensor<6xi1>
  func.return
}

// -----

func.func @compare_op_test_f8e8m0fnu_totalorder_lt() {
  %lhs = stablehlo.constant dense<[0x00, 0x80, 0xFE]> : tensor<3xf8E8M0FNU>
  %rhs = stablehlo.constant dense<[0x01, 0x81, 0xFF]> : tensor<3xf8E8M0FNU>
  %result = stablehlo.compare LT, %lhs, %rhs, TOTALORDER : (tensor<3xf8E8M0FNU>, tensor<3xf8E8M0FNU>) -> tensor<3xi1>
  check.expect_eq_const %result, dense<[true, true, true]> : tensor<3xi1>
  func.return
}

// -----

func.func @compare_op_test_c128_default() {
  // (+NaN, +0.0)
  // (+NaN, -0.0)
  %lhs = stablehlo.constant dense<(0x7FF0000000000001, 0x0000000000000000)> : tensor<complex<f64>>
  %rhs = stablehlo.constant dense<(0x7FF0000000000001, 0x8000000000000000)> : tensor<complex<f64>>
  %result = stablehlo.compare EQ, %lhs, %rhs : (tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<i1>
  check.expect_eq_const %result, dense<false> : tensor<i1>
  func.return
}

// -----

func.func @compare_op_test_c128() {
  // (+NaN, +0.0), (+0.0, +NaN), (-0.0, +0.0), (2.0, 2.0)
  // (+NaN, -0.0), (-0.0, +NaN), (+0.0, +0.0), (2.0, 1.0)
  %lhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x0000000000000000), (0x0000000000000000, 0x7FF0000000000001), (0x8000000000000000, 0x0000000000000000), (2.0, 2.0)]> : tensor<4xcomplex<f64>>
  %rhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x8000000000000000), (0x8000000000000000, 0x7FF0000000000001), (0x0000000000000000, 0x0000000000000000), (2.0, 1.0)]> : tensor<4xcomplex<f64>>
  %result = stablehlo.compare EQ, %lhs, %rhs, FLOAT : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, false, true, false]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare_op_test_c128() {
  // (+NaN, +0.0), (+0.0, +NaN), (-0.0, +0.0), (2.0, 2.0)
  // (+NaN, -0.0), (-0.0, +NaN), (+0.0, +0.0), (2.0, 1.0)
  %lhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x0000000000000000), (0x0000000000000000, 0x7FF0000000000001), (0x8000000000000000, 0x0000000000000000), (2.0, 2.0)]> : tensor<4xcomplex<f64>>
  %rhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x8000000000000000), (0x8000000000000000, 0x7FF0000000000001), (0x0000000000000000, 0x0000000000000000), (2.0, 1.0)]> : tensor<4xcomplex<f64>>
  %result = stablehlo.compare NE, %lhs, %rhs, FLOAT : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[true, true, false, true]> : tensor<4xi1>
  func.return
}
