// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: clamp_op_test_i1
func.func @clamp_op_test_i1() -> tensor<3xi1> {
  %min = stablehlo.constant dense<[false, false, true]> : tensor<3xi1>
  %operand = stablehlo.constant dense<[false, true, false]> : tensor<3xi1>
  %max = stablehlo.constant dense<[true, true, true]> : tensor<3xi1>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<3xi1>, tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>
  func.return %result : tensor<3xi1>
  // CHECK-NEXT: tensor<3xi1>
  // CHECK-NEXT: false
  // CHECK-NEXT: true
  // CHECK-NEXT: true
}

// -----

// CHECK-LABEL: Evaluated results of function: clamp_op_test_si64
func.func @clamp_op_test_si64() -> tensor<3xi64> {
  %min = stablehlo.constant dense<[1, 5, -5]> : tensor<3xi64>
  %operand = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = stablehlo.constant dense<[3, 7, -3]> : tensor<3xi64>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  func.return %result : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: -3 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: clamp_op_test_si64_scalar
func.func @clamp_op_test_si64_scalar() -> tensor<3xi64> {
  %min = stablehlo.constant dense<0> : tensor<i64>
  %operand = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = stablehlo.constant dense<1> : tensor<i64>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<i64>, tensor<3xi64>, tensor<i64>) -> tensor<3xi64>
  func.return %result : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 0 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: clamp_op_test_f64
func.func @clamp_op_test_f64() -> tensor<3xf64> {
  %min = stablehlo.constant dense<[0.0, 0.7, 0xFFF0000000000001]> : tensor<3xf64>
  %operand = stablehlo.constant dense<[0.0, 0.3, 0xFFF0000000000003]> : tensor<3xf64>
  %max = stablehlo.constant dense<[-0.0, 1.0, 0xFFF0000000000002]> : tensor<3xf64>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  func.return %result : tensor<3xf64>
  // CHECK-NEXT: tensor<3xf64>
  // CHECK-NEXT: -0.000000e+00 : f64
  // CHECK-NEXT: 0.69999999999999996 : f64
  // CHECK-NEXT: 0xFFF0000000000003 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: clamp_op_test_c128
func.func @clamp_op_test_c128() -> tensor<3xcomplex<f64>> {
  %min = stablehlo.constant dense<[(1.5, 2.5), (7.5, 5.5), (10.0, 10.0)]> : tensor<3xcomplex<f64>>
  %operand = stablehlo.constant dense<[(2.0, 0.0), (7.5, -5.5), (20.0, 100.0)]> : tensor<3xcomplex<f64>>
  %max = stablehlo.constant dense<[(2.5, 3.5), (7.5, 6.6), (20.0, 20.0)]> : tensor<3xcomplex<f64>>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> tensor<3xcomplex<f64>>
  func.return %result : tensor<3xcomplex<f64>>
  // CHECK-NEXT: tensor<3xcomplex<f64>>
  // CHECK-NEXT: [2.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [7.500000e+00 : f64, 5.500000e+00 : f64]
  // CHECK-NEXT: [2.000000e+01 : f64, 2.000000e+01 : f64]
}
