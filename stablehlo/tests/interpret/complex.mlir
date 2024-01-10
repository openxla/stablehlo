// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @complex_op_test_f64() {
  %lhs = stablehlo.constant dense<[1.0, 3.0]> : tensor<2xf64>
  %rhs = stablehlo.constant dense<[2.0, 4.0]> : tensor<2xf64>
  %result = stablehlo.complex %lhs, %rhs : tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(1.0, 2.0), (3.0, 4.0)]> : tensor<2xcomplex<f64>>
  func.return
}
