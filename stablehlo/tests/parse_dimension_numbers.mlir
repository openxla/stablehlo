// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s
// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | stablehlo-opt -split-input-file | FileCheck %s

// Zero is a required integer value and must survive printing and reparsing.
// CHECK-LABEL: module @zero_index_vector_dim
// CHECK-SAME: stablehlo.gather = #stablehlo.gather<index_vector_dim = 0>
// CHECK-SAME: stablehlo.scatter = #stablehlo.scatter<index_vector_dim = 0>
module @zero_index_vector_dim attributes {
  stablehlo.gather = #stablehlo.gather<index_vector_dim = 0,>,
  stablehlo.scatter = #stablehlo.scatter<index_vector_dim = 0>
} {}

// -----

// Empty arrays may still be omitted.
// CHECK-LABEL: module @empty_arrays
// CHECK-SAME: stablehlo.conv = #stablehlo.conv<[b, f]x[i, o]->[b, f]>
// CHECK-SAME: stablehlo.dot = #stablehlo.dot<>
module @empty_arrays attributes {
  stablehlo.conv = #stablehlo.conv<raw
    output_feature_dimension = 1,
    input_batch_dimension = 0,
    kernel_output_feature_dimension = 1,
    output_batch_dimension = 0,
    input_feature_dimension = 1,
    kernel_input_feature_dimension = 0>,
  stablehlo.dot = #stablehlo.dot<>
} {}

// -----

// expected-error@+2 {{missing required `index_vector_dim` entry}}
// expected-error@+1 {{failed parsing gather dimension numbers attribute}}
module attributes {stablehlo.gather = #stablehlo.gather<>} {}

// -----

module attributes {
  stablehlo.gather = #stablehlo.gather<
    offset_dims = [2, 3],
    collapsed_slice_dims = [0],
    start_index_map = [1, 0]
  // expected-error@+2 {{missing required `index_vector_dim` entry}}
  // expected-error@+1 {{failed parsing gather dimension numbers attribute}}
  >} {}

// -----

// expected-error@+2 {{missing required `index_vector_dim` entry}}
// expected-error@+1 {{failed parsing scatter dimension numbers attribute}}
module attributes {stablehlo.scatter = #stablehlo.scatter<>} {}

// -----

module attributes {
  stablehlo.scatter = #stablehlo.scatter<
    update_window_dims = [1],
    inserted_window_dims = [0, 1],
  // expected-error@+2 {{missing required `index_vector_dim` entry}}
  // expected-error@+1 {{failed parsing scatter dimension numbers attribute}}
  >} {}

// -----

module attributes {
  stablehlo.conv = #stablehlo.conv<raw
    input_feature_dimension = 1,
    kernel_input_feature_dimension = 0,
    kernel_output_feature_dimension = 1,
    output_batch_dimension = 0,
    output_feature_dimension = 1
  // expected-error@+2 {{missing required `input_batch_dimension` entry}}
  // expected-error@+1 {{failed parsing convolution dimension numbers attribute}}
  >} {}

// -----

module attributes {
  stablehlo.conv = #stablehlo.conv<raw
    input_batch_dimension = 0,
    kernel_input_feature_dimension = 0,
    kernel_output_feature_dimension = 1,
    output_batch_dimension = 0,
    output_feature_dimension = 1
  // expected-error@+2 {{missing required `input_feature_dimension` entry}}
  // expected-error@+1 {{failed parsing convolution dimension numbers attribute}}
  >} {}

// -----

module attributes {
  stablehlo.conv = #stablehlo.conv<raw
    input_batch_dimension = 0,
    input_feature_dimension = 1,
    kernel_output_feature_dimension = 1,
    output_batch_dimension = 0,
    output_feature_dimension = 1
  // expected-error@+2 {{missing required `kernel_input_feature_dimension` entry}}
  // expected-error@+1 {{failed parsing convolution dimension numbers attribute}}
  >} {}

// -----

module attributes {
  stablehlo.conv = #stablehlo.conv<raw
    input_batch_dimension = 0,
    input_feature_dimension = 1,
    kernel_input_feature_dimension = 0,
    output_batch_dimension = 0,
    output_feature_dimension = 1
  // expected-error@+2 {{missing required `kernel_output_feature_dimension` entry}}
  // expected-error@+1 {{failed parsing convolution dimension numbers attribute}}
  >} {}

// -----

module attributes {
  stablehlo.conv = #stablehlo.conv<raw
    input_batch_dimension = 0,
    input_feature_dimension = 1,
    kernel_input_feature_dimension = 0,
    kernel_output_feature_dimension = 1,
    output_feature_dimension = 1
  // expected-error@+2 {{missing required `output_batch_dimension` entry}}
  // expected-error@+1 {{failed parsing convolution dimension numbers attribute}}
  >} {}

// -----

module attributes {
  stablehlo.conv = #stablehlo.conv<raw
    input_batch_dimension = 0,
    input_feature_dimension = 1,
    kernel_input_feature_dimension = 0,
    kernel_output_feature_dimension = 1,
    output_batch_dimension = 0
  // expected-error@+2 {{missing required `output_feature_dimension` entry}}
  // expected-error@+1 {{failed parsing convolution dimension numbers attribute}}
  >} {}
