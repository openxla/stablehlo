// RUN: stablehlo-opt %s -split-input-file | FileCheck %s
// RUN: stablehlo-opt %s -split-input-file | stablehlo-opt -split-input-file | FileCheck %s

// Convolution dimension numbers with an out-of-range index cannot be printed in
// the compressed form. The printer must fall back to the raw form instead of
// aborting, and the result must round-trip.

// CHECK: #stablehlo.conv<raw
// CHECK-SAME: output_batch_dimension = 99
module attributes {
  stablehlo.conv = #stablehlo.conv<raw
    input_batch_dimension = 0,
    input_feature_dimension = 3,
    input_spatial_dimensions = [1, 2],
    kernel_input_feature_dimension = 2,
    kernel_output_feature_dimension = 3,
    kernel_spatial_dimensions = [0, 1],
    output_batch_dimension = 99,
    output_feature_dimension = 3,
    output_spatial_dimensions = [2, 1]>
} {}

// -----

// A negative dimension index also falls back to the raw form.
// CHECK: #stablehlo.conv<raw
// CHECK-SAME: input_batch_dimension = -5
module attributes {
  stablehlo.conv = #stablehlo.conv<raw
    input_batch_dimension = -5,
    input_feature_dimension = 3,
    input_spatial_dimensions = [1, 2],
    kernel_input_feature_dimension = 2,
    kernel_output_feature_dimension = 3,
    kernel_spatial_dimensions = [0, 1],
    output_batch_dimension = 0,
    output_feature_dimension = 3,
    output_spatial_dimensions = [1, 2]>
} {}

// -----

// A valid attribute is still printed in the compressed form.
// CHECK: #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>
module attributes {
  stablehlo.conv = #stablehlo.conv<raw
    input_batch_dimension = 0,
    input_feature_dimension = 3,
    input_spatial_dimensions = [1, 2],
    kernel_input_feature_dimension = 2,
    kernel_output_feature_dimension = 3,
    kernel_spatial_dimensions = [0, 1],
    output_batch_dimension = 0,
    output_feature_dimension = 3,
    output_spatial_dimensions = [1, 2]>
} {}
