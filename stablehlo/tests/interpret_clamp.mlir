// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: clamp_op_test_si4
func.func @clamp_op_test_si4() -> tensor<3xi4> {
  %0 = stablehlo.constant dense<[1, 5, -5]> : tensor<3xi4>
  %1 = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi4>
  %2 = stablehlo.constant dense<[3, 7, -3]> : tensor<3xi4>
  %3 = stablehlo.clamp %0, %1, %2 : (tensor<3xi4>, tensor<3xi4>, tensor<3xi4>) -> tensor<3xi4>
  func.return %3 : tensor<3xi4>
  // CHECK-NEXT: tensor<3xi4>
  // CHECK-NEXT: 2 : i4
  // CHECK-NEXT: 5 : i4
  // CHECK-NEXT: -3 : i4
}

// -----

// CHECK-LABEL: Evaluated results of function: clamp_op_test_si4_scalar
func.func @clamp_op_test_si4_scalar() -> tensor<3xi4> {
  %0 = stablehlo.constant dense<0> : tensor<i4>
  %1 = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi4>
  %2 = stablehlo.constant dense<1> : tensor<i4>
  %3 = stablehlo.clamp %0, %1, %2 : (tensor<i4>, tensor<3xi4>, tensor<i4>) -> tensor<3xi4>
  func.return %3 : tensor<3xi4>
  // CHECK-NEXT: tensor<3xi4>
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 0 : i4
}

// -----

// CHECK-LABEL: Evaluated results of function: clamp_op_test_i1
func.func @clamp_op_test_i1() -> tensor<3xi1> {
  %0 = stablehlo.constant dense<[false, false, true]> : tensor<3xi1>
  %1 = stablehlo.constant dense<[false, true, false]> : tensor<3xi1>
  %2 = stablehlo.constant dense<[true, true, true]> : tensor<3xi1>
  %3 = stablehlo.clamp %0, %1, %2 : (tensor<3xi1>, tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>
  func.return %3 : tensor<3xi1>
  // CHECK-NEXT: tensor<3xi1>
  // CHECK-NEXT: false
  // CHECK-NEXT: true
  // CHECK-NEXT: true
}

// -----

// CHECK-LABEL: Evaluated results of function: clamp_op_test_f16
func.func @clamp_op_test_f16() -> tensor<3xf16> {
  %0 = stablehlo.constant dense<[0.0, 0.7, 0x7C80]> : tensor<3xf16>
  %1 = stablehlo.constant dense<[0.0, 0.3, 0x9C80]> : tensor<3xf16>
  %2 = stablehlo.constant dense<[-0.0, 1.0, 0x8C80]> : tensor<3xf16>
  %3 = stablehlo.clamp %0, %1, %2 : (tensor<3xf16>, tensor<3xf16>, tensor<3xf16>) -> tensor<3xf16>
  func.return %3 : tensor<3xf16>
  // CHECK-NEXT: tensor<3xf16>
  // CHECK-NEXT: -0.000000e+00 : f16
  // CHECK-NEXT: 7.001950e-01 : f16
  // CHECK-NEXT: 0x7C80 : f16
}

// -----

// CHECK-LABEL: Evaluated results of function: clamp_op_test_c64
func.func @clamp_op_test_c64() -> tensor<3xcomplex<f32>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (7.5, 5.5), (10.0, 10.0)]> : tensor<3xcomplex<f32>>
  %1 = stablehlo.constant dense<[(2.0, 0.0), (7.5, -5.5), (20.0, 100.0)]> : tensor<3xcomplex<f32>>
  %2 = stablehlo.constant dense<[(2.5, 3.5), (7.5, 6.6), (20.0, 20.0)]> : tensor<3xcomplex<f32>>
  %3 = stablehlo.clamp %0, %1, %2 : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>>
  func.return %3 : tensor<3xcomplex<f32>>
  // CHECK-NEXT: tensor<3xcomplex<f32>>
  // CHECK-NEXT: [2.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [7.500000e+00 : f32, 5.500000e+00 : f32]
  // CHECK-NEXT: [2.000000e+01 : f32, 2.000000e+01 : f32]
}
