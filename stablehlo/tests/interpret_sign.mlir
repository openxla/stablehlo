// RUN: stablehlo-translate --interpret -split-input-file %s

// CHECK-LABEL: Evaluated results of function: sign_op_test_si64
func.func @sign_op_test_si64() -> tensor<3xi64> {
  %operand = stablehlo.constant dense<[-1, 0, 1]> : tensor<3xi64>
  %result = stablehlo.sign %operand : tensor<3xi64>
  func.return %result : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 1 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_f64
func.func @sign_op_test_f64() -> tensor<5xf64> {
  // +NaN, -1.0, -0.0, +0.0, 1.0
  %operand = stablehlo.constant dense<[0x7FFFFFFFFFFFFFFF, -1.0, -0.0, 0.0, 1.0]> : tensor<5xf64>
  %result = stablehlo.sign %operand : tensor<5xf64>
  func.return %result : tensor<5xf64>
  // CHECK-NEXT: tensor<5xf64>
  // CHECK-NEXT: 0xFFFFFFFFFFFFFFFF : f64
  // CHECK-NEXT: -1.000000e+00 : f64
  // CHECK-NEXT: -0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_c128
func.func @sign_op_test_c128() -> tensor<3xcomplex<f64>> {
  // (+NaN, +0.0), (+0.0, +NaN), (0.0, 1.0)
  // (+NaN, +0.0), (+Nan, +0.0), (1.0, 1.0)
  %operand = stablehlo.constant dense<[(0x7FF0000000000001, 0x0000000000000000), (0x0000000000000000, 0x7FF0000000000001), (0.0, 1.0)]> : tensor<3xcomplex<f64>>
  %result = stablehlo.sign %operand : tensor<3xcomplex<f64>>
  func.return %result : tensor<3xcomplex<f64>>
  // CHECK-NEXT: tensor<3xcomplex<f64>>
  // CHECK-NEXT: [0x7FF0000000000001, 0x0000000000000000]
  // CHECK-NEXT: [0x7FF0000000000001, 0x0000000000000000]
  // CHECK-NEXT: [0.000000e+00, 1.000000e+00]
}
