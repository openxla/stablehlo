// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.25.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.25.0}}
module {
  func.func @dot_general_block_scaling(%arg0: tensor<2x4x4xf32>, %arg1: tensor<2x4x4xf32>, %scale: tensor<2x4x1xf32>) -> tensor<2x4x4xf32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.dot_general_v3' that was explicitly marked illegal}}
    %0 = "stablehlo.dot_general"(%arg0, %arg1, %scale) {
      dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      block_scaling_config = #stablehlo.block_scaling_config<lhs = #stablehlo.tensor_block_scaling_config<scale_idx = 2>>
    } : (tensor<2x4x4xf32>, tensor<2x4x4xf32>, tensor<2x4x1xf32>) -> tensor<2x4x4xf32>
    func.return %0 : tensor<2x4x4xf32>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.25.0}}
module {
  func.func @dot_general_sparsity(%arg0: tensor<2x4x2xf32>, %arg1: tensor<2x8x4xf32>, %meta: tensor<2x4x2xi16>) -> tensor<2x4x4xf32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.dot_general_v3' that was explicitly marked illegal}}
    %0 = "stablehlo.dot_general"(%arg0, %arg1, %meta) {
      dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      sparsity_config = #stablehlo.sparsity_config<lhs = <num_non_zero = 1, block_size = 4, dimension = 2, stride = 1, idx = 2>>
    } : (tensor<2x4x2xf32>, tensor<2x8x4xf32>, tensor<2x4x2xi16>) -> tensor<2x4x4xf32>
    func.return %0 : tensor<2x4x4xf32>
  }
}
