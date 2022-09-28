// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: transpose_op_test_si32
func.func @transpose_op_test_si32() -> tensor<3x2x2xi32> {
  %0 = stablehlo.constant dense<[[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]> : tensor<2x3x2xi32>
  %1 = "stablehlo.transpose"(%0) {permutation = dense<[1,0,2]> : tensor<3xi64>} : (tensor<2x3x2xi32>) -> tensor<3x2x2xi32>
  return %1 : tensor<3x2x2xi32>
  // CHECK-NEXT: tensor<3x2x2xi32>
  // CHECK-NEXT:  1 : i32
  // CHECK-NEXT:  2 : i32
  // CHECK-NEXT:  7 : i32
  // CHECK-NEXT:  8 : i32
  // CHECK-NEXT:  3 : i32
  // CHECK-NEXT:  4 : i32
  // CHECK-NEXT:  9 : i32
  // CHECK-NEXT:  10 : i32
  // CHECK-NEXT:  5 : i32
  // CHECK-NEXT:  6 : i32
  // CHECK-NEXT:  11 : i32
  // CHECK-NEXT:  12 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: transpose_op_test_si32
func.func @transpose_op_test_si32() -> tensor<2x3x2xi32> {
  %0 = stablehlo.constant dense<[[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]> : tensor<2x3x2xi32>
  %1 = "stablehlo.transpose"(%0) {permutation = dense<[2,1,0]> : tensor<3xi64>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  return %1 : tensor<2x3x2xi32>
  // CHECK-NEXT: tensor<2x3x2xi32> {
  // CHECK-NEXT:  1 : i32
  // CHECK-NEXT:  7 : i32
  // CHECK-NEXT:  3 : i32
  // CHECK-NEXT:  9 : i32
  // CHECK-NEXT:  5 : i32
  // CHECK-NEXT:  11 : i32
  // CHECK-NEXT:  2 : i32
  // CHECK-NEXT:  8 : i32
  // CHECK-NEXT:  4 : i32
  // CHECK-NEXT:  10 : i32
  // CHECK-NEXT:  6 : i32
  // CHECK-NEXT:  12 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: transpose_op_test_si32
func.func @transpose_op_test_si32() -> tensor<2x3x2xi32> {
  %0 = stablehlo.constant dense<[[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]> : tensor<2x3x2xi32>
  %1 = "stablehlo.transpose"(%0) {permutation = dense<[2,1,0]> : tensor<3xi64>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  %2 = "stablehlo.transpose"(%1) {permutation = dense<[2,1,0]> : tensor<3xi64>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  return %2 : tensor<2x3x2xi32>
  // CHECK-NEXT: tensor<2x3x2xi32> {
  // CHECK-NEXT:  1 : i32
  // CHECK-NEXT:  2 : i32
  // CHECK-NEXT:  3 : i32
  // CHECK-NEXT:  4 : i32
  // CHECK-NEXT:  5 : i32
  // CHECK-NEXT:  6 : i32
  // CHECK-NEXT:  7 : i32
  // CHECK-NEXT:  8 : i32
  // CHECK-NEXT:  9 : i32
  // CHECK-NEXT:  10 : i32
  // CHECK-NEXT:  11 : i32
  // CHECK-NEXT:  12 : i32
}
