// RUN: stablehlo-translate --interpret --split-input-file %s | FileCheck %s
// RUN: stablehlo-translate --interpret --interpreter-print-dense --split-input-file %s | FileCheck %s --check-prefix=CHECK-DENSE

func.func @test_ranks_and_dtypes() -> (tensor<4xi8>, tensor<4xf16>, tensor<2xi1>, tensor<ui8>, tensor<1xi8>, tensor<2x2xi8>, tensor<2x2x2xi64>, tensor<0xui8>, tensor<0x2xi8>, tensor<2x0xi8>) {
  // CHECK:      tensor<4xi8> {
  // CHECK-NEXT:   [1, 2, -3, -4]
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<[1, 2, -3, -4]> : tensor<4xi8>
  %ints = stablehlo.constant dense<[1, 2, -3, -4]> : tensor<4xi8>

  // CHECK:      tensor<4xf16> {
  // CHECK-NEXT:   [1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00]
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<[1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00]> : tensor<4xf16>
  %floats = stablehlo.constant dense<[1.0, 2.0, -3.0, -4.0]> : tensor<4xf16>

  // CHECK:      tensor<2xi1> {
  // CHECK-NEXT:   [true, false]
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<[true, false]> : tensor<2xi1>
  %bool = stablehlo.constant dense<[true, false]> : tensor<2xi1>

  // CHECK:      tensor<ui8> {1}
  // CHECK-DENSE: dense<1> : tensor<ui8>
  %scalar = stablehlo.constant dense<1> : tensor<ui8>

  // CHECK:      tensor<1xi8> {
  // CHECK-NEXT:   [1]
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<1> : tensor<1xi8>
  %array = stablehlo.constant dense<[1]> : tensor<1xi8>

  // CHECK:      tensor<2x2xi8> {
  // CHECK-NEXT:   [
  // CHECK-NEXT:     [1, 2],
  // CHECK-NEXT:     [3, 4]
  // CHECK-NEXT:   ]
  // CHECK-NEXT: }
  // CHECK-DENSE{LITERAL}: dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>
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
  // CHECK-DENSE{LITERAL}: dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>
  %tensor = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>

  // CHECK:      tensor<0xui8> {
  // CHECK-NEXT:   []
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<> : tensor<0xui8>
  %empty_scalar = stablehlo.constant dense<> : tensor<0xui8>

  // CHECK:      tensor<0x2xi8> {
  // CHECK-NEXT:   []
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<> : tensor<0x2xi8>
  %empty_matrix = stablehlo.constant dense<> : tensor<0x2xi8>

  // CHECK:      tensor<2x0xi8> {
  // CHECK-NEXT:   [
  // CHECK-NEXT:     [],
  // CHECK-NEXT:     []
  // CHECK-NEXT:   ]
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<> : tensor<2x0xi8>
  %empty_matrix_col = stablehlo.constant dense<> : tensor<2x0xi8>

  func.return %ints, %floats, %bool, %scalar, %array, %matrix, %tensor, %empty_scalar, %empty_matrix, %empty_matrix_col : tensor<4xi8>, tensor<4xf16>, tensor<2xi1>, tensor<ui8>, tensor<1xi8>, tensor<2x2xi8>, tensor<2x2x2xi64>, tensor<0xui8>, tensor<0x2xi8>, tensor<2x0xi8>
}

// -----

func.func @test_sub_byte() -> (tensor<2xi2>, tensor<4xi4>, tensor<4xf6E2M3FN>, tensor<4xf8E3M4>) {
  // CHECK:      tensor<2xi2> {
  // CHECK-NEXT:   [1, -2]
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<[1, -2]> : tensor<2xi2>
  %i2 = stablehlo.constant dense<[1, -2]> : tensor<2xi2>

  // CHECK:      tensor<4xi4> {
  // CHECK-NEXT:  [1, 2, 3, 4]
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<[1, 2, 3, 4]> : tensor<4xi4>
  %i4 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi4>

  // CHECK:      tensor<4xf6E2M3FN> {
  // CHECK-NEXT:  [1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00]
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<[1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00]> : tensor<4xf6E2M3FN>
  %f6 = stablehlo.constant dense<[1.0, 2.0, -3.0, -4.0]> : tensor<4xf6E2M3FN>

  // CHECK:      tensor<4xf8E3M4> {
  // CHECK-NEXT:  [1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00]
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<[1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00]> : tensor<4xf8E3M4>
  %f8 = stablehlo.constant dense<[1.0, 2.0, -3.0, -4.0]> : tensor<4xf8E3M4>
  func.return %i2, %i4, %f6, %f8 : tensor<2xi2>, tensor<4xi4>, tensor<4xf6E2M3FN>, tensor<4xf8E3M4>
}

// -----

func.func @test_complex() -> (tensor<4xcomplex<f32>>, tensor<4xcomplex<f64>>) {
  // CHECK:      tensor<4xcomplex<f32>> {
  // CHECK-NEXT{LITERAL}:   [[3.140000e+00, 1.000000e+00], [3.140000e+00, 1.000000e+00], [3.140000e+00, 1.000000e+00], [3.140000e+00, 1.000000e+00]]
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<(3.140000e+00,1.000000e+00)> : tensor<4xcomplex<f32>>
  %complex_f32 = stablehlo.constant dense<(3.140000e+00, 1.0)> : tensor<4xcomplex<f32>>

  // CHECK:      tensor<4xcomplex<f64>> {
  // CHECK-NEXT{LITERAL}:   [[3.140000e+00, 1.000000e+00], [3.140000e+00, 1.000000e+00], [3.140000e+00, 1.000000e+00], [3.140000e+00, 1.000000e+00]]
  // CHECK-NEXT: }
  // CHECK-DENSE: dense<(3.140000e+00,1.000000e+00)> : tensor<4xcomplex<f64>>
  %complex_f64 = stablehlo.constant dense<(3.140000e+00, 1.0)> : tensor<4xcomplex<f64>>
  func.return %complex_f32, %complex_f64 : tensor<4xcomplex<f32>>, tensor<4xcomplex<f64>>
}
