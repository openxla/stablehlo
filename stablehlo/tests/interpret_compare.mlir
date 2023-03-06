// RUN: stablehlo-interpreter --interpret -split-input-file %s

func.func @compare() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare EQ, %lhs, %rhs, SIGNED : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[true, false, true, false, true]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare NE, %lhs, %rhs, SIGNED : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[false, true, false, true, false]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare GE, %lhs, %rhs, SIGNED : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[true, true, true, true, true]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare GT, %lhs, %rhs, SIGNED : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[false, true, false, true, false]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare LE, %lhs, %rhs, SIGNED : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[true, false, true, false, true]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>
  %result = stablehlo.compare LT, %lhs, %rhs, SIGNED : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
  check.expect_eq_const %result, dense<[false, false, false, false, false]> : tensor<5xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare EQ, %lhs, %rhs, UNSIGNED : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[true, false]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare NE, %lhs, %rhs, UNSIGNED : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[false, true]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare GE, %lhs, %rhs, UNSIGNED : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[true, true]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare GT, %lhs, %rhs, UNSIGNED : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[false, true]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare LE, %lhs, %rhs, UNSIGNED : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[true, false]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[0, 1]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[0, 0]> : tensor<2xui64>
  %result = stablehlo.compare LT, %lhs, %rhs, UNSIGNED : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
  check.expect_eq_const %result, dense<[false, false]> : tensor<2xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare EQ, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[true, false, false, true]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare NE, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, true, true, false]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare GE, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[true, true, false, true]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare GT, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, true, false, false]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare LE, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[true, false, true, true]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  %lhs = stablehlo.constant dense<[true, true, false, false]> : tensor<4xi1>
  %rhs = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  %result = stablehlo.compare LT, %lhs, %rhs, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, false, true, false]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare EQ, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, false, true, false, true, false, true, true, true, false, true, true, false, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare NE, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[true, true, false, true, false, true, false, false, false, true, false, false, true, true]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare GE, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, false, true, false, true, false, true, true, true, false, true, true, false, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare GT, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare LE, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, false, true, true, true, true, true, true, true, true, true, true, false, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare LT, %lhs, %rhs, FLOAT : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, false, false, true, false, true, false, false, false, true, false, false, false, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare() {
  // (+NaN, +0.0), (+0.0, +NaN), (-0.0, +0.0), (2.0, 2.0)
  // (+NaN, -0.0), (-0.0, +NaN), (+0.0, +0.0), (2.0, 1.0)
  %lhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x0000000000000000), (0x0000000000000000, 0x7FF0000000000001), (0x8000000000000000, 0x0000000000000000), (2.0, 2.0)]> : tensor<4xcomplex<f64>>
  %rhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x8000000000000000), (0x8000000000000000, 0x7FF0000000000001), (0x0000000000000000, 0x0000000000000000), (2.0, 1.0)]> : tensor<4xcomplex<f64>>
  %result = stablehlo.compare EQ, %lhs, %rhs, FLOAT : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, false, true, false]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  // (+NaN, +0.0), (+0.0, +NaN), (-0.0, +0.0), (2.0, 2.0)
  // (+NaN, -0.0), (-0.0, +NaN), (+0.0, +0.0), (2.0, 1.0)
  %lhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x0000000000000000), (0x0000000000000000, 0x7FF0000000000001), (0x8000000000000000, 0x0000000000000000), (2.0, 2.0)]> : tensor<4xcomplex<f64>>
  %rhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x8000000000000000), (0x8000000000000000, 0x7FF0000000000001), (0x0000000000000000, 0x0000000000000000), (2.0, 1.0)]> : tensor<4xcomplex<f64>>
  %result = stablehlo.compare NE, %lhs, %rhs, FLOAT : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[true, true, false, true]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  // (+NaN, +0.0), (+0.0, +NaN), (-0.0, +0.0), (2.0, 2.0)
  // (+NaN, -0.0), (-0.0, +NaN), (+0.0, +0.0), (2.0, 1.0)
  %lhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x0000000000000000), (0x0000000000000000, 0x7FF0000000000001), (0x8000000000000000, 0x0000000000000000), (2.0, 2.0)]> : tensor<4xcomplex<f64>>
  %rhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x8000000000000000), (0x8000000000000000, 0x7FF0000000000001), (0x0000000000000000, 0x0000000000000000), (2.0, 1.0)]> : tensor<4xcomplex<f64>>
  %result = stablehlo.compare GE, %lhs, %rhs, FLOAT : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, false, true, true]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  // (+NaN, +0.0), (+0.0, +NaN), (-0.0, +0.0), (2.0, 2.0)
  // (+NaN, -0.0), (-0.0, +NaN), (+0.0, +0.0), (2.0, 1.0)
  %lhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x0000000000000000), (0x0000000000000000, 0x7FF0000000000001), (0x8000000000000000, 0x0000000000000000), (2.0, 2.0)]> : tensor<4xcomplex<f64>>
  %rhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x8000000000000000), (0x8000000000000000, 0x7FF0000000000001), (0x0000000000000000, 0x0000000000000000), (2.0, 1.0)]> : tensor<4xcomplex<f64>>
  %result = stablehlo.compare GT, %lhs, %rhs, FLOAT : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, false, false, true]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  // (+NaN, +0.0), (+0.0, +NaN), (-0.0, +0.0), (2.0, 2.0)
  // (+NaN, -0.0), (-0.0, +NaN), (+0.0, +0.0), (2.0, 1.0)
  %lhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x0000000000000000), (0x0000000000000000, 0x7FF0000000000001), (0x8000000000000000, 0x0000000000000000), (2.0, 2.0)]> : tensor<4xcomplex<f64>>
  %rhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x8000000000000000), (0x8000000000000000, 0x7FF0000000000001), (0x0000000000000000, 0x0000000000000000), (2.0, 1.0)]> : tensor<4xcomplex<f64>>
  %result = stablehlo.compare LE, %lhs, %rhs, FLOAT : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, false, true, false]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  // (+NaN, +0.0), (+0.0, +NaN), (-0.0, +0.0), (2.0, 2.0)
  // (+NaN, -0.0), (-0.0, +NaN), (+0.0, +0.0), (2.0, 1.0)
  %lhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x0000000000000000), (0x0000000000000000, 0x7FF0000000000001), (0x8000000000000000, 0x0000000000000000), (2.0, 2.0)]> : tensor<4xcomplex<f64>>
  %rhs = stablehlo.constant dense<[(0x7FF0000000000001, 0x8000000000000000), (0x8000000000000000, 0x7FF0000000000001), (0x0000000000000000, 0x0000000000000000), (2.0, 1.0)]> : tensor<4xcomplex<f64>>
  %result = stablehlo.compare LT, %lhs, %rhs, FLOAT : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xi1>
  check.expect_eq_const %result, dense<[false, false, false, false]> : tensor<4xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare EQ, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[true, false, true, false, true, false, true, false, true, false, true, true, true, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare NE, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, true, false, true, false, true, false, true, false, true, false, false, false, true]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare GE, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[true, false, true, false, true, false, true, false, true, false, true, true, true, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare GT, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare LE, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<14xi1>
  func.return
}

// -----

func.func @compare() {
  // -NaN, -NaN, -Inf, -Inf, -2.0, -2.0, -0.0, -0.0, +0.0, 1.0, 2.0, +Inf, +NaN, +minNaN
  // -NaN, +NaN, -Inf, +Inf, -2.0, -1.0, -0.0, +0.0, +0.0, 2.0, 2.0, +Inf, +NaN, +maxNaN
  %lhs = stablehlo.constant dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.0, -2.0, 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 1.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>
  %rhs = stablehlo.constant dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.0, -1.0, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 2.0, 2.0, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>
  %result = stablehlo.compare LT, %lhs, %rhs, TOTALORDER : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
  check.expect_eq_const %result, dense<[false, true, false, true, false, true, false, true, false, true, false, false, false, true]> : tensor<14xi1>
  func.return
}
