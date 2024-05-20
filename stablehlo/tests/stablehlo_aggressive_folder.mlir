// RUN: stablehlo-opt --stablehlo-aggressive-folder --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: func @eval_iota
func.func @eval_iota() -> (tensor<3x4x5xi32>, tensor<3x4x5xi32>, tensor<3x4x5xi32>) {
  // CHECK-NOT: stablehlo.iota
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
  // CHECK-SAME: {{\[}}[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
  // CHECK-SAME: {{\[}}[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]]> : tensor<3x4x5xi32>

  // CHECK: [[RESULT1:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
  // CHECK-SAME: {{\[}}[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
  // CHECK-SAME: {{\[}}[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]]> : tensor<3x4x5xi32>

  // CHECK: [[RESULT2:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
  // CHECK-SAME: {{\[}}[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
  // CHECk-SAME: {{\[}}[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]]> : tensor<3x4x5xi32>

  // CHECK: return [[RESULT0]], [[RESULT1]], [[RESULT2]]
  %0 = stablehlo.iota dim = 0 : tensor<3x4x5xi32>
  %1 = stablehlo.iota dim = 1 : tensor<3x4x5xi32>
  %2 = stablehlo.iota dim = 2 : tensor<3x4x5xi32>
  func.return %0, %1, %2 : tensor<3x4x5xi32>, tensor<3x4x5xi32>, tensor<3x4x5xi32>
}

// -----

// CHECK-LABEL: func @eval_iota_zero_dimension
func.func @eval_iota_zero_dimension() -> (tensor<0xi32>, tensor<5x0x2xi32>) {
  // CHECK-NOT: stablehlo.iota
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<> : tensor<0xi32>
  // CHECK: [[RESULT1:%.*]] = stablehlo.constant dense<> : tensor<5x0x2xi32>
  // CHECK: return [[RESULT0]], [[RESULT1]]
  %0 = stablehlo.iota dim = 0 : tensor<0xi32>
  %1 = stablehlo.iota dim = 2 : tensor<5x0x2xi32>
  func.return %0, %1 : tensor<0xi32>, tensor<5x0x2xi32>
}
