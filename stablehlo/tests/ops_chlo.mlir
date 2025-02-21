// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s


// CHECK-LABEL: func @broadcast_add_quantized
func.func @broadcast_add_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>> {
  %0 = "chlo.broadcast_add"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
}

// -----

func.func @broadcast_add_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>> {
  // expected-error @+1{{'chlo.broadcast_add' op requires compatible element types for all operands and results}}
  %0 = "chlo.broadcast_add"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
}

// -----

// CHECK-LABEL: func @broadcast_max_quantized
func.func @broadcast_max_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>> {
  %0 = "chlo.broadcast_maximum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
}

// -----

func.func @broadcast_max_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>> {
  // expected-error @+1{{'chlo.broadcast_maximum' op requires compatible element types for all operands and results}}
  %0 = "chlo.broadcast_maximum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
}

// -----

func.func @broadcast_max_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f16, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>> {
  // expected-error @+1{{'chlo.broadcast_maximum' op requires compatible element types for all operands and results}}
  %0 = "chlo.broadcast_maximum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f16, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
}

// -----

// CHECK-LABEL: func @broadcast_min_quantized
func.func @broadcast_min_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>> {
  %0 = "chlo.broadcast_minimum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
}

// -----

func.func @broadcast_min_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>> {
  // expected-error @+1{{'chlo.broadcast_minimum' op requires compatible element types for all operands and results}}
  %0 = "chlo.broadcast_minimum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 3.0:15>>) -> tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i16:f32, 2.0:15>>
}

// -----

func.func @broadcast_min_quantized(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f16, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>> {
  // expected-error @+1{{'chlo.broadcast_minimum' op requires compatible element types for all operands and results}}
  %0 = "chlo.broadcast_minimum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f16, 3.0:15>>) -> tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>
}

// -----

func.func @constant_like(%arg0: tensor<1x2xi64>) -> (tensor<1x2xi32>) {
  // expected-error @+1 {{'chlo.constant_like' op value's type doesn't match element return type}}
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<1x2xi64>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

// ragged_dot mode 1: [b,m,k], [g,b,k,n], [g] -> [b,m,n]
func.func @ragged_dot_non_contracting(%lhs : tensor<2x11x5xf32>, %rhs : tensor<3x2x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<2x11x7xf32> {
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2],
      lhs_ragged_dimensions = [1],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  func.return %0 : tensor<2x11x7xf32>
}

// -----

// ragged_dot mode 2: [m,k], [k,n], [g] -> [g,m,n]
func.func @ragged_dot_contracting(%lhs : tensor<2x11x5xf32>, %rhs : tensor<2x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<3x2x11x7xf32> {
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [2],
      rhs_group_dimensions = []
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<2x11x5xf32>, tensor<2x5x7xf32>, tensor<3xi64>) -> tensor<3x2x11x7xf32>
  func.return %0 : tensor<3x2x11x7xf32>
}

// -----

// ragged_dot mode 3: [b,m,k], [b,k,n], [g] -> [b,m,n]
func.func @ragged_dot_batch(%lhs : tensor<3x11x5xf32>, %rhs : tensor<3x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<3x11x7xf32> {
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [0],
      rhs_group_dimensions = []
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<3x11x5xf32>, tensor<3x5x7xf32>, tensor<3xi64>) -> tensor<3x11x7xf32>
  func.return %0 : tensor<3x11x7xf32>
}

// -----

func.func @ragged_dot_incompatible_contracting_dims(%lhs : tensor<11x5xf32>, %rhs : tensor<3x2x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<11x7xf32> {
  // @expected-error@+1 {{contracting dimension sizes must match}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [0],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<11x5xf32>, tensor<3x2x7xf32>, tensor<3xi64>) -> tensor<11x7xf32>
  func.return %0 : tensor<11x7xf32>
}

// -----

func.func @ragged_dot_group_sizes_incorrect_rank(%lhs : tensor<11x5xf32>, %rhs : tensor<3x5x7xf32>, %group_sizes : tensor<3x2xi64>) -> tensor<11x7xf32> {
  // @expected-error@+1 {{expected group_sizes to have rank 1, got 2}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [0],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<11x5xf32>, tensor<3x5x7xf32>, tensor<3x2xi64>) -> tensor<11x7xf32>
  func.return %0 : tensor<11x7xf32>
}

// -----

func.func @ragged_dot_mode1_group_sizes_broadcasted(%lhs : tensor<19x17x11x5xf32>, %rhs : tensor<3x19x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<19x17x11x7xf32> {
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2],
      lhs_ragged_dimensions = [2],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<19x17x11x5xf32>, tensor<3x19x5x7xf32>, tensor<3xi64>) -> tensor<19x17x11x7xf32>
  func.return %0 : tensor<19x17x11x7xf32>
}

// -----

func.func @ragged_dot_mode1_group_sizes_incorrect_shape(%lhs : tensor<19x17x11x5xf32>, %rhs : tensor<3x19x5x7xf32>, %group_sizes : tensor<19x11x3xi64>) -> tensor<19x17x11x7xf32> {
  // @expected-error@+1 {{group_sizes is expected to have shape [19, 17, 3], got [19, 11, 3]}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2],
      lhs_ragged_dimensions = [2],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<19x17x11x5xf32>, tensor<3x19x5x7xf32>, tensor<19x11x3xi64>) -> tensor<19x17x11x7xf32>
  func.return %0 : tensor<19x17x11x7xf32>
}

// -----

func.func @ragged_dot_mode2_group_sizes_incorrect_shape(%lhs : tensor<19x11x17x5xf32>, %rhs : tensor<19x17x5x7xf32>, %group_sizes : tensor<19x11x3xi64>) -> tensor<3x19x11x7xf32> {
  // @expected-error@+1 {{group_sizes is expected to have shape [19, 17, 3], got [19, 11, 3]}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2,3],
      rhs_contracting_dimensions = [1,2],
      lhs_ragged_dimensions = [3],
      rhs_group_dimensions = []
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<19x11x17x5xf32>, tensor<19x17x5x7xf32>, tensor<19x11x3xi64>) -> tensor<3x19x11x7xf32>
  func.return %0 : tensor<3x19x11x7xf32>
}

// -----

func.func @ragged_dot_mode3_group_sizes_incorrect_shape(%lhs : tensor<17x19x11x5xf32>, %rhs : tensor<17x19x5x7xf32>, %group_sizes : tensor<19x3xi64>) -> tensor<17x19x11x7xf32> {
  // @expected-error@+1 {{group_sizes is expected to have shape [17, 3], got [19, 3]}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0,1],
      rhs_batching_dimensions = [0,1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2],
      lhs_ragged_dimensions = [1],
      rhs_group_dimensions = []
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<17x19x11x5xf32>, tensor<17x19x5x7xf32>, tensor<19x3xi64>) -> tensor<17x19x11x7xf32>
  func.return %0 : tensor<17x19x11x7xf32>
}

// -----

func.func @ragged_dot_incorrect_group_dim_size(%lhs : tensor<11x5xf32>, %rhs : tensor<3x5x7xf32>, %group_sizes : tensor<2xi64>) -> tensor<11x7xf32> {
  // @expected-error@+1 {{rhs group dimension is expected to have size=2, got 3}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [0],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<11x5xf32>, tensor<3x5x7xf32>, tensor<2xi64>) -> tensor<11x7xf32>
  func.return %0 : tensor<11x7xf32>
}

// -----

func.func @ragged_dot_incorrect_number_of_lhs_ragged_dimensions(%lhs : tensor<11x5xf32>, %rhs : tensor<3x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<11x7xf32> {
  // @expected-error@+1 {{There must be exactly one ragged dimension in the lhs}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [0, 1],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<11x5xf32>, tensor<3x5x7xf32>, tensor<3xi64>) -> tensor<11x7xf32>
  func.return %0 : tensor<11x7xf32>
}

// -----

func.func @ragged_dot_rhs_group_dim_is_batch(%lhs : tensor<3x11x5xf32>, %rhs : tensor<3x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<3x11x7xf32> {
  // @expected-error@+1 {{has duplicated dimension from rhs_group_dimensions and rhs_batching_dimensions: 0}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [1],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<3x11x5xf32>, tensor<3x5x7xf32>, tensor<3xi64>) -> tensor<3x11x7xf32>
  func.return %0 : tensor<3x11x7xf32>
}

// -----

func.func @ragged_dot_rhs_group_dim_is_contracting(%lhs : tensor<11x3xf32>, %rhs : tensor<3x3x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<11x7xf32> {
  // @expected-error@+1 {{has duplicated dimension from rhs_group_dimensions and rhs_contracting_dimensions: 1}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [0],
      rhs_group_dimensions = [1]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<11x3xf32>, tensor<3x3x7xf32>, tensor<3xi64>) -> tensor<11x7xf32>
  func.return %0 : tensor<11x7xf32>
}

// -----

func.func @ragged_dot_nonzero_rhs_group_dims_for_ragged_batch(%lhs : tensor<2x11x5xf32>, %rhs : tensor<3x2x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<2x11x7xf32> {
  // @expected-error@+1 {{There must be zero group dimensions in the rhs when the ragged dimension is batch or contracting}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2],
      lhs_ragged_dimensions = [0],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  func.return %0 : tensor<2x11x7xf32>
}

// -----

func.func @ragged_dot_nonzero_rhs_group_dims_for_ragged_contracting(%lhs : tensor<11x5xf32>, %rhs : tensor<3x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<11x7xf32> {
  // @expected-error@+1 {{There must be zero group dimensions in the rhs when the ragged dimension is batch or contracting}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [1],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<11x5xf32>, tensor<3x5x7xf32>, tensor<3xi64>) -> tensor<11x7xf32>
  func.return %0 : tensor<11x7xf32>
}

// -----

func.func @ragged_dot_zero_rhs_group_dims_for_ragged_noncontracting(%lhs : tensor<11x5xf32>, %rhs : tensor<5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<11x7xf32> {
  // @expected-error@+1 {{There must be exactly one group dimension in the rhs when the lhs ragged dimension is non-contracting}}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0],
      lhs_ragged_dimensions = [0],
      rhs_group_dimensions = []
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<11x5xf32>, tensor<5x7xf32>, tensor<3xi64>) -> tensor<11x7xf32>
  func.return %0 : tensor<11x7xf32>
}

// -----

func.func @top_k(%arg0 : tensor<f32>) {
  // expected-error @+2 {{failed to infer returned types}}
  // @expected-error @+1{{operand's rank must be at least 1}}
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<f32> -> (tensor<8xf32>, tensor<8xi32>)
  return
}

// -----

func.func @top_k(%arg0 : tensor<4xf32>) {
  // expected-error @+2 {{failed to infer returned types}}
  // @expected-error @+1{{operand's last dimension must be at least 8}}
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<4xf32> -> (tensor<8xf32>, tensor<8xi32>)
  return
}

// -----

func.func @top_k_1d(%arg0 : tensor<16xf32>) {
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<16xf32> -> (tensor<8xf32>, tensor<8xi32>)
  return
}

// -----

func.func @top_k_nd(%arg0 : tensor<16x16xf32>) {
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<16x16xf32> -> (tensor<16x8xf32>, tensor<16x8xi32>)
  return
}

// -----

func.func @top_k_unbounded(%arg0 : tensor<?x16x?xf32>) {
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<?x16x?xf32> -> (tensor<?x16x8xf32>, tensor<?x16x8xi32>)
  return
}

// -----

func.func @top_k_bounded(%arg0 : tensor<?x?x?xf32, #stablehlo.bounds<?, 16, 16>>) {
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<?x?x?xf32, #stablehlo.bounds<?, 16, 16>> -> (tensor<16x?x8xf32, #stablehlo.bounds<?, 16, ?>>, tensor<16x?x8xi32, #stablehlo.bounds<?, 16, ?>>)
  return
}

// -----

func.func @top_k_unranked(%arg0 : tensor<*xf32>) {
  %0:2 = chlo.top_k(%arg0, k=8) : tensor<*xf32> -> (tensor<*xf32>, tensor<*xi32>)
  return
}

// -----

func.func @erf_inv(%arg0 : tensor<16x16xf32>) {
  %0 = chlo.erf_inv %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
  return
}
