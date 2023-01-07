// RUN: stablehlo-opt --inline %s | FileCheck %s

// CHECK: func.func @main
func.func @main(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: chlo.broadcast_add
  %0 = func.call @callee(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-NOT: func.func private @callee
func.func private @callee(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<f32> {
  %0 = "chlo.broadcast_add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
