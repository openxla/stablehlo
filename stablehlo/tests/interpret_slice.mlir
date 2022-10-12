// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: slice_op
func.func @slice_op() -> tensor<2x2xi64> {
  %0 = stablehlo.constant dense<[[0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1]]> : tensor<3x6xi64>
  %1 = "stablehlo.slice"(%0) {
    start_indices = dense<[0, 2]> : tensor<2xi64>,
    limit_indices = dense<[3, 6]> : tensor<2xi64>,
    strides = dense<[2, 3]> : tensor<2xi64>
  } : (tensor<3x6xi64>) -> tensor<2x2xi64>
  func.return %1 : tensor<2x2xi64>
  // CHECK-NEXT: tensor<2x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
}
