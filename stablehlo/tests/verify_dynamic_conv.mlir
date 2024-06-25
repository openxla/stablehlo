// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: func @dynamic_conv
func.func @dynamic_conv(%arg0 : tensor<100x26x26x32xf32>, %arg1 : tensor<3x3x1x32xf32>) ->
    tensor<100x28x28x1xf32> {
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi64>) ->
       tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

// CHECK: func @dynamic_conv_empty_spatial_dimensions
// CHECK: stablehlo.dynamic_conv
// CHECK: batch_group_count = 1 : i64,
// CHECK: dimension_numbers = #stablehlo.conv<[b, f]x[i, o]->[b, f]>,
// CHECK: feature_group_count = 1 : i64
func.func @dynamic_conv_empty_spatial_dimensions(%arg0: tensor<3x2xf16>,
    %arg1: tensor<2x2xf16>) -> tensor<3x2xf16> {
  %padding = stablehlo.constant dense<0> : tensor<0x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, f]x[i, o]->[b, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<3x2xf16>, tensor<2x2xf16>, tensor<0x2xi64>) -> tensor<3x2xf16>
  func.return %result : tensor<3x2xf16>
}

// -----

// CHECK-LABEL: func @dynamic_conv_upcast
func.func @dynamic_conv_upcast(%arg0 : tensor<100x26x26x32xi8>,
    %arg1 : tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32> {
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xi8>, tensor<3x3x1x32xi8>, tensor<2x2xi64>) ->
       tensor<100x28x28x1xi32>
  func.return %result : tensor<100x28x28x1xi32>
}

// -----

func.func @dynamic_conv_c1(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects convolution arguments to have same number of dimensions. Got: 'tensor<1x8x8x207xf32>' and 'tensor<3x3x207xf32>'.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c2(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects window-strides to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    window_strides = array<i64: 1>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c3(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects window to have positive stride for 1-th window dimension, but got 0.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    window_strides = array<i64: 1, 0>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c4_invalid_padding_dim_0(%arg0 : tensor<100x26x26x32xf32>, %arg1 : tensor<3x3x1x32xf32>) ->
    tensor<100x28x28x1xf32> {
  // expected-error@+2 {{expects padding to be of shape [2, 2], but got [3, 2]}}
  %padding = stablehlo.constant dense<2> : tensor<3x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<3x2xi64>) ->
       tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @dynamic_conv_c4_invalid_padding_dim_1(%arg0 : tensor<100x26x26x32xf32>, %arg1 : tensor<3x3x1x32xf32>) ->
    tensor<100x28x28x1xf32> {
  // expected-error@+2 {{expects padding to be of shape [2, 2], but got [2, 3]}}
  %padding = stablehlo.constant dense<2> : tensor<2x3xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x3xi64>) ->
       tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @dynamic_conv_c5(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects base-dilation factors to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    lhs_dilation = array<i64: 1>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c6(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects window to have positive base dilation factor for 0-th window dimension, but got 0.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    lhs_dilation = array<i64: 0, 1>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c7(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects window-dilation factors to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    rhs_dilation = array<i64: 1>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c8(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects window to have positive window dilation factor for 0-th window dimension, but got 0.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    rhs_dilation = array<i64: 0, 1>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c9(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects window-reversal to have same dimension-size as size of window dimensions (2), but got: 1.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    window_reversal = array<i1: false>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c10(%arg0: tensor<5x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects input batch dimension (5) to be divisible by batch_group_count. Got batch_group_count = 2.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 2 : i64
  } : (tensor<5x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c11(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x20x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects input feature dimension (207) to be a multiple of feature_group_count. Got feature_group_count = 2.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 2 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x20x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c12(%arg0: tensor<1x8x8x32x207xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<32x1x8x8x16xf32> {
  // expected-error@+2{{expects convolution arguments to have 4 dimensions. Got: 5}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 4,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 4,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 1,
      output_feature_dimension = 4,
      output_spatial_dimensions = [2, 3]
    >,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x32x207xf32>, tensor<3x3x32x207x16xf32>, tensor<2x2xi64>) ->
       tensor<32x1x8x8x16xf32>
  func.return %result : tensor<32x1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c13(%arg0: tensor<1xf32>, %arg1: tensor<3xf32>)
    -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects convolution arguments to have >= 2 dimensions. Got: 'tensor<1xf32>' and 'tensor<3xf32>'.}}
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1xf32>, tensor<3xf32>, tensor<2x2xi64>) -> tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c13(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+2 {{expects input dimension-numbers to be unique, got {0, 0, 1, 2}.}}
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 0,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi64>) ->
    tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @dynamic_conv_c13(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+2 {{expects input, kernel, and output dimension-numbers to be in-range [0, 4).}}
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = -1,
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
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi64>) ->
    tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @dynamic_conv_c13(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+2 {{expects input, kernel, and output dimension-numbers to be in-range [0, 4).}}
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 4,
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
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi64>) ->
    tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @dynamic_conv_c14(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x20x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects input feature dimension (207) / feature_group_count = kernel input feature dimension (20). Got feature_group_count = 1.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x20x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c15(%arg0: tensor<3x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<3x8x8x16xf32> {
  // expected-error@+2 {{expects output feature dimension size (16) to be a multiple of batch_group_count. Got batch_group_count = 3.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 3 : i64
  } : (tensor<3x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<3x8x8x16xf32>
  func.return %result : tensor<3x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c16(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x69x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects kernel output feature dimension (16) to be divisible by feature_group_count. For feature_group_count = 3.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 3 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x69x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c17(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects the same size for input, kernel and output spatial-dimensions, but got 2, 3, and 2 resp.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, 2, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c18(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+2 {{expects kernel dimension-numbers to be unique, got {3, 2, 0, 0}.}}
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 0],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi64>) ->
       tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @dynamic_conv_c18(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+2 {{expects input, kernel, and output dimension-numbers to be in-range [0, 4).}}
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = -1,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi64>) ->
       tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @dynamic_conv_c18(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+2 {{expects input, kernel, and output dimension-numbers to be in-range [0, 4).}}
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 4,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi64>) ->
    tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @dynamic_conv_c19(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects the same size for input, kernel and output spatial-dimensions, but got 2, 2, and 3 resp.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, 2, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c20(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+2 {{expects output dimension-numbers to be unique, got {0, 3, 0, 3}.}}
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [0, 3]
    >,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi64>) ->
       tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @dynamic_conv_c20(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+2 {{expects input, kernel, and output dimension-numbers to be in-range [0, 4).}}
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = -1,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi64>) ->
       tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @dynamic_conv_c20(%arg0 : tensor<100x26x26x32xf32>,
    %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  // expected-error@+2 {{expects input, kernel, and output dimension-numbers to be in-range [0, 4).}}
  %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 4,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi64>) ->
       tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

func.func @dynamic_conv_c21(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{op attribute 'feature_group_count' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 0 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c22(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{op attribute 'batch_group_count' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 0 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c23(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects batch_group_count and feature_group_count not to be both greater than 1. Got 2 and 2 resp.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 2 : i64,
    batch_group_count = 2 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}

// -----

func.func @dynamic_conv_c24(%arg0: tensor<3x2xf16>, %arg1: tensor<2x2xf16>) -> tensor<3x2xf16> {
  // expected-error@+2{{expects precision config to be empty or have <= 2 elements}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers =  #stablehlo.conv<[b, f]x[i, o]->[b, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<3x2xf16>, tensor<2x2xf16>, tensor<2x2xi64>) -> tensor<3x2xf16>
  func.return %result : tensor<3x2xf16>
}

// -----

func.func @dynamic_conv_c27(%arg0: tensor<1x4x4x1xi64>,
    %arg1: tensor<3x3x1x1xi32>) -> tensor<1x2x2x1xi64> {
  // expected-error@+2 {{expects lhs and rhs to have compatible element type. Got: 'i64' and 'i32'}}
    %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
    %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    window_strides = array<i64: 4, 4>,
    lhs_dilation = array<i64: 2, 2>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi32>, tensor<2x2xi64>) ->
       tensor<1x2x2x1xi64>
  func.return %result : tensor<1x2x2x1xi64>
}

// -----

func.func @dynamic_conv_invalid_window_attributes(%arg0: tensor<1x8x8x207xf32>,
    %arg1: tensor<0x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error@+2 {{expects window to have positive value for 0-th window dimension, but got 0.}}
  %padding = stablehlo.constant dense<0> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%arg0, %arg1, %padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<0x3x207x16xf32>, tensor<2x2xi64>) ->
       tensor<1x8x8x16xf32>
  func.return %result : tensor<1x8x8x16xf32>
}
