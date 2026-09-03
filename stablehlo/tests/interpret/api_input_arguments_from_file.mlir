// RUN: echo "[dense<1> : tensor<2xi32>, dense<2> : tensor<2xi32>]" > %t.args
// RUN: stablehlo-translate %s --interpret --args=@%t.args | FileCheck %s

// RUN: not stablehlo-translate %s --interpret --args=@%t.does_not_exist 2>&1 | FileCheck %s --check-prefixes=CHECK-ERROR-NO-FILE
// CHECK-ERROR-NO-FILE: failed to read args file

// RUN: not stablehlo-translate %s --interpret --args=@%t.args,@%t.args 2>&1 | FileCheck %s --check-prefixes=CHECK-ERROR-MULTIPLE
// CHECK-ERROR-MULTIPLE: expected a single args file, multiple files are only supported for .npy args

func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK:      tensor<2xi32> {
// CHECK-NEXT:   [3, 3]
// CHECK-NEXT: }
