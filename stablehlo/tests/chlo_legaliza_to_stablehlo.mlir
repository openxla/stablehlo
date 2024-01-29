// RUN:stablehlo-opt -chlo-legalize-to-stablehlo="legalize-broadcasts=true expand-compositions=false" -cse -canonicalize -split-input-file -verify-diagnostics %s -o - | FileCheck %s

// Check the non-broadcast case for each registered op, then just check a
// representative op for detailed broadcast semantics.
// CHECK-LABEL: @addWithoutBroadcast
func.func @addWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stablehlo.add %arg0, %arg1
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
