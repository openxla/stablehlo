// RUN: stablehlo-opt --stablehlo-target-independent-optimization --split-input-file %s | FileCheck %s

// Check that simplificaiton and folding are both applied.

// CHECK-LABEL: @add_cst_on_rhs
func.func @add_cst_on_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<1.0> : tensor<f32>
  %0 = stablehlo.add %cst, %cst : tensor<f32>
  // CHECK: stablehlo.add %arg0, %cst : tensor<f32>
  %1 = stablehlo.add %0, %arg0 : tensor<f32>
  return %1 : tensor<f32>
}
