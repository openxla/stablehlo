// RUN: stablehlo-opt --stablehlo-legalize-composite-to-call --split-input-file %s | FileCheck %s
// RUN: stablehlo-opt --stablehlo-legalize-composite-to-call=except='foo.baz,foo.qux' --split-input-file %s | FileCheck %s --check-prefix=EXCEPT

// CHECK-LABEL: func @composite
// EXCEPT-LABEL: func @composite
func.func @composite(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  // CHECK-NEXT: call @bar(%arg0, %arg1)
  // CHECK-NOT: stablehlo.composite
  // EXCEPT-NEXT: call @bar(%arg0, %arg1)
  // EXCEPT-NOT: stablehlo.composite
  %0 = stablehlo.composite "foo.bar" %arg0, %arg1 {
    decomposition = @bar
  } : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK: call @baz(%arg0)
  // CHECK-NOT: stablehlo.composite
  // EXCEPT-NEXT: stablehlo.composite "foo.baz"
  // EXCEPT-NOT: call
  %1 = stablehlo.composite "foo.baz" %arg0 {
    decomposition = @baz
  } : (tensor<4xf32>) -> tensor<4xf32>
  return
}

func.func @bar(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}

func.func @baz(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}
