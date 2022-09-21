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

// -----

// CHECK-LABEL: Evaluated results of function: transpose_op_test_si32
func.func @transpose_op_test_si32() -> tensor<2x3x2x3xi32> {
  %0 = stablehlo.constant dense<[[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]], [[[19,20,21],[22,23,24]],[[25,26,27],[28,29,30]],[[31,32,33],[34,35,36]]]]> : tensor<2x3x2x3xi32>
  %1 = "stablehlo.transpose"(%0) {permutation = dense<[3,1,2,0]> : tensor<4xi64>} : (tensor<2x3x2x3xi32>) -> tensor<3x3x2x2xi32>
  %2 = "stablehlo.transpose"(%1) {permutation = dense<[0,2,1,3]> : tensor<4xi64>} : (tensor<3x3x2x2xi32>) -> tensor<3x2x3x2xi32>
  %3 = "stablehlo.transpose"(%2) {permutation = dense<[3,2,1,0]> : tensor<4xi64>} : (tensor<3x2x3x2xi32>) -> tensor<2x3x2x3xi32>
  return %3 : tensor<2x3x2x3xi32>
  // CHECK-NEXT: tensor<2x3x2x3xi32> {
  // CHECK-NEXT:   1 : i32
  // CHECK-NEXT:   2 : i32
  // CHECK-NEXT:   3 : i32
  // CHECK-NEXT:   4 : i32
  // CHECK-NEXT:   5 : i32
  // CHECK-NEXT:   6 : i32
  // CHECK-NEXT:   7 : i32
  // CHECK-NEXT:   8 : i32
  // CHECK-NEXT:   9 : i32
  // CHECK-NEXT:   10 : i32
  // CHECK-NEXT:   11 : i32
  // CHECK-NEXT:   12 : i32
  // CHECK-NEXT:   13 : i32
  // CHECK-NEXT:   14 : i32
  // CHECK-NEXT:   15 : i32
  // CHECK-NEXT:   16 : i32
  // CHECK-NEXT:   17 : i32
  // CHECK-NEXT:   18 : i32
  // CHECK-NEXT:   19 : i32
  // CHECK-NEXT:   20 : i32
  // CHECK-NEXT:   21 : i32
  // CHECK-NEXT:   22 : i32
  // CHECK-NEXT:   23 : i32
  // CHECK-NEXT:   24 : i32
  // CHECK-NEXT:   25 : i32
  // CHECK-NEXT:   26 : i32
  // CHECK-NEXT:   27 : i32
  // CHECK-NEXT:   28 : i32
  // CHECK-NEXT:   29 : i32
  // CHECK-NEXT:   30 : i32
  // CHECK-NEXT:   31 : i32
  // CHECK-NEXT:   32 : i32
  // CHECK-NEXT:   33 : i32
  // CHECK-NEXT:   34 : i32
  // CHECK-NEXT:   35 : i32
  // CHECK-NEXT:   36 : i32
}
