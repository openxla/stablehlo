// RUN: stablehlo-translate %s --interpret --args="[dense<1> : tensor<2xi32>, dense<2> : tensor<2xi32>]" | FileCheck %s

// RUN: not stablehlo-translate %s --interpret --args="not_array" 2>&1 | FileCheck %s --check-prefixes=CHECK-ERROR-NOT-ARRAY
// CHECK-ERROR-NOT-ARRAY: expectected array attribute string for args, i.e. --args=[dense<1> : tensor<2xi32>, ...]

// RUN: not stablehlo-translate %s --interpret --args="[4.0 : f32]" 2>&1 | FileCheck %s --check-prefixes=CHECK-ERROR-NOT-DENSE
// CHECK-ERROR-NOT-DENSE: expected dense elements attribute for args elements, i.e. --args=[dense<1> : tensor<2xi32>, ...]

func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK:      tensor<2xi32> {
// CHECK-NEXT:   [3, 3]
// CHECK-NEXT: }
