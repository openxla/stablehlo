// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: select_op_test_si64
func.func @select_op_test_si64() -> tensor<3xi64> {
  %pred = stablehlo.constant dense<[true, false, true]> : tensor<3xi1>
  %on_true = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %on_false = stablehlo.constant dense<[3, 7, -3]> : tensor<3xi64>
  %result = stablehlo.select %pred, %on_true, %on_false : (tensor<3xi1>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  func.return %result : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 7 : i64
  // CHECK-NEXT: -1 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: select_op_test_si64_scalar
func.func @select_op_test_si64_scalar() -> tensor<3xi64> {
  %pred = stablehlo.constant dense<false> : tensor<i1>
  %on_true = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %on_false = stablehlo.constant dense<[3, 7, -3]> : tensor<3xi64>
  %result = stablehlo.select %pred, %on_true, %on_false : (tensor<i1>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  func.return %result : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 7 : i64
  // CHECK-NEXT: -3 : i64
}
