// RUN: stablehlo-translate --interpret %s | FileCheck %s

func.func @main() -> (tensor<4xi8>, tensor<4xf16>, tensor<2xi1>, tensor<ui8>, tensor<1xi8>, tensor<2x2xi8>, tensor<2x2x2xi64>, tensor<0xui8>, tensor<0x2xi8>, tensor<2x0xi8>) {
  // CHECK:      tensor<4xi8> {
  // CHECK-NEXT:   [1, 2, -3, -4]
  // CHECK-NEXT: }
  %ints = stablehlo.constant dense<[1, 2, -3, -4]> : tensor<4xi8>

  // CHECK:      tensor<4xf16> {
  // CHECK-NEXT:   [1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00]
  // CHECK-NEXT: }
  %floats = stablehlo.constant dense<[1.0, 2.0, -3.0, -4.0]> : tensor<4xf16>

  // CHECK:      tensor<2xi1> {
  // CHECK-NEXT:   [true, false]
  // CHECK-NEXT: }
  %bool = stablehlo.constant dense<[true, false]> : tensor<2xi1>

  // CHECK:      tensor<ui8> {1}
  %scalar = stablehlo.constant dense<1> : tensor<ui8>

  // CHECK:      tensor<1xi8> {
  // CHECK-NEXT:   [1]
  // CHECK-NEXT: }
  %array = stablehlo.constant dense<[1]> : tensor<1xi8>

  // CHECK:      tensor<2x2xi8> {
  // CHECK-NEXT:   [
  // CHECK-NEXT:     [1, 2],
  // CHECK-NEXT:     [3, 4]
  // CHECK-NEXT:   ]
  // CHECK-NEXT: }
  %matrix = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>

  // CHECK:      tensor<2x2x2xi64> {
  // CHECK-NEXT:   [
  // CHECK-NEXT:     [
  // CHECK-NEXT:       [0, 1],
  // CHECK-NEXT:       [2, 3]
  // CHECK-NEXT:     ],
  // CHECK-NEXT:     [
  // CHECK-NEXT:       [4, 0],
  // CHECK-NEXT:       [1, 2]
  // CHECK-NEXT:     ]
  // CHECK-NEXT:   ]
  // CHECK-NEXT: }
  %tensor = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>

  // CHECK:      tensor<0xui8> {
  // CHECK-NEXT:   []
  // CHECK-NEXT: }
  %empty_scalar = stablehlo.constant dense<> : tensor<0xui8>

  // CHECK:      tensor<0x2xi8> {
  // CHECK-NEXT:   []
  // CHECK-NEXT: }
  %empty_matrix = stablehlo.constant dense<> : tensor<0x2xi8>

  // CHECK:      tensor<2x0xi8> {
  // CHECK-NEXT:   [
  // CHECK-NEXT:     [],
  // CHECK-NEXT:     []
  // CHECK-NEXT:   ]
  // CHECK-NEXT: }
  %empty_matrix_col = stablehlo.constant dense<> : tensor<2x0xi8>

  func.return %ints, %floats, %bool, %scalar, %array, %matrix, %tensor, %empty_scalar, %empty_matrix, %empty_matrix_col : tensor<4xi8>, tensor<4xf16>, tensor<2xi1>, tensor<ui8>, tensor<1xi8>, tensor<2x2xi8>, tensor<2x2x2xi64>, tensor<0xui8>, tensor<0x2xi8>, tensor<2x0xi8>
}
