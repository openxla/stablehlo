// RUN: stablehlo-translate --interpret -split-input-file %s

// CHECK-LABEL: Evaluated results of function: is_finite_op_test_bf16
func.func @is_finite_op_test_bf16() -> tensor<8xi1> {
  // -0.0, +0.0, -3.14, +3.14, -NaN, +NaN, -Inf, +Inf
  %operand = stablehlo.constant dense<[-0.0, 0.0, -3.140625, 3.140625, 0xFFFF, 0x7FFF, 0xFF80, 0x7F80]> : tensor<8xbf16>
  %result = stablehlo.is_finite %operand : (tensor<8xbf16>) -> tensor<8xi1>
  func.return %result : tensor<8xi1>
  // CHECK-NEXT: tensor<8xi1>
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
}

// -----

// CHECK-LABEL: Evaluated results of function: is_finite_op_test_f16
func.func @is_finite_op_test_f16() -> tensor<8xi1> {
  // -0.0, +0.0, -3.14, +3.14, -NaN, +NaN, -Inf, +Inf
  %operand = stablehlo.constant dense<[-0.0, 0.0, -3.141, 3.141, 0xFFFF, 0x7FFF, 0xFF80, 0x7F80]> : tensor<8xf16>
  %result = stablehlo.is_finite %operand : (tensor<8xf16>) -> tensor<8xi1>
  func.return %result : tensor<8xi1>
  // CHECK-NEXT: tensor<8xi1>
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
}

// -----

// CHECK-LABEL: Evaluated results of function: is_finite_op_test_f32
func.func @is_finite_op_test_f32() -> tensor<8xi1> {
  // -0.0, +0.0, -3.14, +3.14, -NaN, +NaN, -Inf, +Inf
  %operand = stablehlo.constant dense<[-0.0, 0.0, -3.14159265, 3.14159265, 0xFFFFFFFF, 0x7FFFFFFF, 0xFF800000, 0x7F800000]> : tensor<8xf32>
  %result = stablehlo.is_finite %operand : (tensor<8xf32>) -> tensor<8xi1>
  func.return %result : tensor<8xi1>
  // CHECK-NEXT: tensor<8xi1>
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
}

// -----

// CHECK-LABEL: Evaluated results of function: is_finite_op_test_f64
func.func @is_finite_op_test_f64() -> tensor<8xi1> {
  // -0.0, +0.0, -3.14, +3.14, -NaN, +NaN, -Inf, +Inf
  %operand = stablehlo.constant dense<[-0.0, 0.0, -3.14159265358979323846, 3.14159265358979323846, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0xFFF0000000000000, 0x7FF0000000000000]> : tensor<8xf64>
  %result = stablehlo.is_finite %operand : (tensor<8xf64>) -> tensor<8xi1>
  func.return %result : tensor<8xi1>
  // CHECK-NEXT: tensor<8xi1>
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: true : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
  // CHECK-NEXT: false : i1
}
