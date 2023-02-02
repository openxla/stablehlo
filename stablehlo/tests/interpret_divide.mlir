// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: div_op_test_ui64
func.func @div_op_test_ui64() -> tensor<4xui64> {
  %0 = stablehlo.constant dense<[17, 18, 19, 20]> : tensor<4xui64>
  %1 = stablehlo.constant dense<[3, 4, 5, 7]> : tensor<4xui64>
  %2 = stablehlo.divide %0, %1 : tensor<4xui64>
  func.return %2 : tensor<4xui64>
  // CHECK-NEXT: tensor<4xui64>
  // CHECK-NEXT: 5 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 2 : ui64
}

// -----

// CHECK-LABEL: Evaluated results of function: div_op_test_si64
func.func @div_op_test_si64() -> tensor<4xi64> {
  %0 = stablehlo.constant dense<[17, -17, 17, -17]> : tensor<4xi64>
  %1 = stablehlo.constant dense<[3, 3, -3, -3]> : tensor<4xi64>
  %2 = stablehlo.divide %0, %1 : tensor<4xi64>
  func.return %2 : tensor<4xi64>
  // CHECK-NEXT: tensor<4xi64>
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: -5 : i64
  // CHECK-NEXT: -5 : i64
  // CHECK-NEXT: 5 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: div_op_test_f64
func.func @div_op_test_f64() -> tensor<4xf64> {
  %0 = stablehlo.constant dense<[17.1, -17.1, 17.1, -17.1]> : tensor<4xf64>
  %1 = stablehlo.constant dense<[3.0, 3.0, -3.0, -3.0]> : tensor<4xf64>
  %2 = stablehlo.divide %0, %1 : tensor<4xf64>
  func.return %2 : tensor<4xf64>
  // CHECK-NEXT: tensor<4xf64>
  // CHECK-NEXT: 5.700000e+00 : f64
  // CHECK-NEXT: -5.700000e+00 : f64
  // CHECK-NEXT: -5.700000e+00 : f64
  // CHECK-NEXT: 5.700000e+00 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: div_op_test_c64
func.func @div_op_test_c64() -> tensor<2xcomplex<f64>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (7.5, 5.5)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.constant dense<[(2.5, 1.5), (5.5, 7.5)]> : tensor<2xcomplex<f64>>
  %2 = stablehlo.divide %0, %1 : tensor<2xcomplex<f64>>
  func.return %2 : tensor<2xcomplex<f64>>
  // CHECK-NEXT: tensor<2xcomplex<f64>>
  // CHECK-NEXT: [0.88235294117647056 : f64, 0.4705882352941177 : f64]
  // CHECK-NEXT: [0.95375722543352603 : f64, -0.30057803468208094 : f64]
}
