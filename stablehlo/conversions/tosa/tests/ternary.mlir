// RUN: stablehlo-opt %s --stablehlo-legalize-to-tosa | FileCheck %s

// CHECK-LABEL: @concatenate
func.func @concatenate(%arg0 : tensor<5x2xf32>, %arg1 : tensor<5x5xf32>, %arg2 : tensor<5x7xf32>) -> tensor<5x14xf32> {
  // CHECK: tosa.concat %arg0, %arg1, %arg2 {axis = 1 : i32} : (tensor<5x2xf32>, tensor<5x5xf32>, tensor<5x7xf32>) -> tensor<5x14xf32>
  %0 = "stablehlo.concatenate"(%arg0, %arg1, %arg2) {dimension = 1 : i64} : (tensor<5x2xf32>, tensor<5x5xf32>, tensor<5x7xf32>) -> tensor<5x14xf32>
  return %0 : tensor<5x14xf32>
}

// CHECK-LABEL: @select
func.func @select(%arg0 : tensor<10xi1>, %arg1 : tensor<10xf32>, %arg2 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.select
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<10xi1>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
