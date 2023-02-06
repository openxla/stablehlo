// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: square_root_op_test_f64
func.func @square_root_op_test_f64() -> tensor<4xf64> {
  %operand = stablehlo.constant dense<[25.33, -36.11, 1.0, 0.0]> : tensor<4xf64>
  %result = stablehlo.sqrt %operand : tensor<4xf64>
  func.return %result : tensor<4xf64>
  // CHECK-NEXT: tensor<4xf64>
  // CHECK-NEXT: 5.0328918128646478 : f64
  // CHECK-NEXT: 0xFFF8000000000000 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: square_root_op_test_c128
func.func @square_root_op_test_c128() -> tensor<2xcomplex<f64>> {
  %operand = stablehlo.constant dense<[(-1.0, 0.0), (3.0, 4.0)]> : tensor<2xcomplex<f64>>
  %result = stablehlo.sqrt %operand : tensor<2xcomplex<f64>>
  func.return %result : tensor<2xcomplex<f64>>
  // CHECK-NEXT: tensor<2xcomplex<f64>>
  // CHECK-NEXT: [0.000000e+00 : f64, 1.000000e+00 : f64]
  // CHECK-NEXT: [2.000000e+00 : f64, 1.000000e+00 : f64]
}
