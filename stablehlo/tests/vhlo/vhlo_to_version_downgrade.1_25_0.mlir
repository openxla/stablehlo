// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.25.0' %s | FileCheck %s

// DotGeneralOp was changed in v1.26.0 to have ext_operands, block_scaling_config and sparsity_config. Ensure that serializing for 1.25.0 is valid and targets the v1.25.0 opset when none of them are used.

// CHECK-LABEL: vhlo.func_v1 @dot_general_plain
func.func @dot_general_plain(%arg0: tensor<2x4x4xf32>, %arg1: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
  // CHECK: vhlo.dot_general_v2
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>
  } : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  func.return %0 : tensor<2x4x4xf32>
}
