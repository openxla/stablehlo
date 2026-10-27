// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.19.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.19.0}}
module {
  func.func @collective_broadcast_has_dynamic_root(%arg0: tensor<16x8xf32>, %arg1: tensor<1xi32>) -> tensor<16x8xf32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.collective_broadcast_v2' that was explicitly marked illegal}}
    %0 = "stablehlo.collective_broadcast"(%arg0, %arg1) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      has_dynamic_root
    } : (tensor<16x8xf32>, tensor<1xi32>) -> tensor<16x8xf32>
    func.return %0 : tensor<16x8xf32>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.19.0}}
module {
  func.func @convolution_mixed_fp8(%arg0: tensor<100x26x26x32xf8E5M2>, %arg1: tensor<3x3x1x32xf8E4M3FN>) -> tensor<100x28x28x1xf8E5M2> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.convolution_v1' that was explicitly marked illegal}}
    %result = "stablehlo.convolution"(%arg0, %arg1) {
      batch_group_count = 1 : i64,
      dimension_numbers = #stablehlo.conv<raw
        input_batch_dimension = 0,
        input_feature_dimension = 3,
        input_spatial_dimensions = [1, 2],
        kernel_input_feature_dimension = 3,
        kernel_output_feature_dimension = 2,
        kernel_spatial_dimensions = [0, 1],
        output_batch_dimension = 0,
        output_feature_dimension = 3,
        output_spatial_dimensions = [1, 2]
      >,
      feature_group_count = 1 : i64,
      lhs_dilation = array<i64: 1, 1>,
      padding = dense<2> : tensor<2x2xi64>,
      rhs_dilation = array<i64: 1, 1>,
      window_strides = array<i64: 1, 1>
    } : (tensor<100x26x26x32xf8E5M2>, tensor<3x3x1x32xf8E4M3FN>) -> tensor<100x28x28x1xf8E5M2>
    func.return %result : tensor<100x28x28x1xf8E5M2>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.19.0}}
module {
  func.func @dynamic_conv_mixed_fp8(%arg0: tensor<100x26x26x32xf8E5M2>, %arg1: tensor<3x3x1x32xf8E4M3FN>, %arg2: tensor<2x2xi64>) -> tensor<100x28x28x1xf8E5M2> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.dynamic_conv_v2' that was explicitly marked illegal}}
    %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %arg2) {
      batch_group_count = 1 : i64,
      dimension_numbers = #stablehlo.conv<raw
        input_batch_dimension = 0,
        input_feature_dimension = 3,
        input_spatial_dimensions = [1, 2],
        kernel_input_feature_dimension = 3,
        kernel_output_feature_dimension = 2,
        kernel_spatial_dimensions = [0, 1],
        output_batch_dimension = 0,
        output_feature_dimension = 3,
        output_spatial_dimensions = [1, 2]
      >,
      feature_group_count = 1 : i64,
      lhs_dilation = array<i64: 1, 1>,
      rhs_dilation = array<i64: 1, 1>,
      window_strides = array<i64: 1, 1>
    } : (tensor<100x26x26x32xf8E5M2>, tensor<3x3x1x32xf8E4M3FN>, tensor<2x2xi64>) -> tensor<100x28x28x1xf8E5M2>
    func.return %result : tensor<100x28x28x1xf8E5M2>
  }
}
