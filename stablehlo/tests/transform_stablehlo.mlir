// RUN: stablehlo-opt --inline %s | FileCheck %s

// CHECK: func.func @main
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: stablehlo.abs
  %0 = func.call @callee(%arg0): (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-NOT: func.func private @callee
func.func private @callee(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.abs %arg0 : tensor<f32>
  func.return %0 : tensor<f32>
}
