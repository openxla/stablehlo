// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: reshape_op_test_si32
func.func @reshape_op_test_si32() -> tensor<6xi32> {
  %0 = stablehlo.constant dense<[[1,2,3,4,5,6]]> : tensor<1x6xi32>
  %1 = stablehlo.reshape %0 : (tensor<1x6xi32>) -> tensor<6xi32>
  func.return %1 : tensor<6xi32>
  // CHECK-NEXT: tensor<6xi32> {
  // CHECK-NEXT:   1 : i32
  // CHECK-NEXT:   2 : i32
  // CHECK-NEXT:   3 : i32
  // CHECK-NEXT:   4 : i32
  // CHECK-NEXT:   5 : i32
  // CHECK-NEXT:   6 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: reshape_op_test_si32
func.func @reshape_op_test_si32() -> tensor<2x3xi32> {
  %0 = stablehlo.constant dense<[1,2,3,4,5,6]> : tensor<6xi32>
  %1 = stablehlo.reshape %0 : (tensor<6xi32>) -> tensor<2x3xi32>
  func.return %1 : tensor<2x3xi32>
  // CHECK-NEXT: tensor<2x3xi32> {
  // CHECK-NEXT:   1 : i32
  // CHECK-NEXT:   2 : i32
  // CHECK-NEXT:   3 : i32
  // CHECK-NEXT:   4 : i32
  // CHECK-NEXT:   5 : i32
  // CHECK-NEXT:   6 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: reshape_op_test_si32
func.func @reshape_op_test_si32() -> tensor<3x2xi32> {
  %0 = stablehlo.constant dense<[[1,2,3],[4,5,6]]> : tensor<2x3xi32>
  %1 = stablehlo.reshape %0 : (tensor<2x3xi32>) -> tensor<3x2xi32>
  func.return %1 : tensor<3x2xi32>
  // CHECK-NEXT: tensor<3x2xi32> {
  // CHECK-NEXT:   1 : i32
  // CHECK-NEXT:   2 : i32
  // CHECK-NEXT:   3 : i32
  // CHECK-NEXT:   4 : i32
  // CHECK-NEXT:   5 : i32
  // CHECK-NEXT:   6 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: reshape_op_test_si32
func.func @reshape_op_test_si32() -> tensor<6xi32> {
  %0 = stablehlo.constant dense<[[1,2],[3,4],[5,6]]> : tensor<3x2xi32>
  %1 = stablehlo.reshape %0 : (tensor<3x2xi32>) -> tensor<6xi32>
  func.return %1 : tensor<6xi32>
  // CHECK-NEXT: tensor<6xi32> {
  // CHECK-NEXT:   1 : i32
  // CHECK-NEXT:   2 : i32
  // CHECK-NEXT:   3 : i32
  // CHECK-NEXT:   4 : i32
  // CHECK-NEXT:   5 : i32
  // CHECK-NEXT:   6 : i32
}
