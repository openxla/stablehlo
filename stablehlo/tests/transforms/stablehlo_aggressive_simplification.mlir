// RUN: stablehlo-opt --stablehlo-aggressive-simplification=fold-op-element-limit=100 --allow-unregistered-dialect --split-input-file %s | FileCheck %s

/////////
// AddOp

// CHECK-LABEL: @add_cst_on_rhs
func.func @add_cst_on_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<1.0> : tensor<f32>
  // CHECK: stablehlo.add %arg0, %cst : tensor<f32>
  %0 = stablehlo.add %cst, %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @add_zero_like_lhs
func.func @add_zero_like_lhs(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.add %0, %arg0 : tensor<i32>
  // CHECK-NOT: stablehlo.constant
  // CHECK: return %arg0
  return %1 : tensor<i32>
}

// CHECK-LABEL: @add_zero_like_rhs
func.func @add_zero_like_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.add %arg0, %0 : tensor<f32>
  // CHECK-NOT: stablehlo.constant
  // CHECK: return %arg0
  return %1 : tensor<f32>
}

// CHECK-LABEL: @add_cst_on_rhs_with_attrs
func.func @add_cst_on_rhs_with_attrs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<1.0> : tensor<f32>
  // CHECK: stablehlo.add %arg0, %cst {mhlo.frontend_attributes = {foo = "1"}} : tensor<f32>
  %0 = stablehlo.add %cst, %arg0 {mhlo.frontend_attributes = {foo = "1"}} : tensor<f32>
  return %0 : tensor<f32>
}

// -----

/////////
// AndOp

// CHECK-LABEL: @and_cst_on_rhs
func.func @and_cst_on_rhs(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %cst = stablehlo.constant dense<true> : tensor<2xi1>
  %0 = stablehlo.and %cst, %arg0 : tensor<2xi1>
  // Check that constant canonicalized to RHS, then other patterns apply
  // CHECK-NOT: stablehlo.and
  // return %arg0
  return %0 : tensor<2xi1>
}

// CHECK-LABEL: @and_zero
func.func @and_zero(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = stablehlo.constant dense<false> : tensor<2xi1>
  %1 = stablehlo.and %0, %arg0 : tensor<2xi1>
  // CHECK-NOT: stablehlo.and
  // CHECK: [[FALSE:%.+]] = stablehlo.constant dense<false> : tensor<2xi1>
  // CHECK: return [[FALSE]]
  return %1 : tensor<2xi1>
}

// CHECK-LABEL: @and_one
func.func @and_one(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = stablehlo.constant dense<true> : tensor<2xi1>
  %1 = stablehlo.and %0, %arg0 : tensor<2xi1>
  // CHECK-NOT: stablehlo.and
  // CHECK: return %arg0
  return %1 : tensor<2xi1>
}

// CHECK-LABEL: @and_i32_one
func.func @and_i32_one(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.constant dense<1> : tensor<2xi32>
  %1 = stablehlo.and %0, %arg0 : tensor<2xi32>
  // CHECK: %[[AND:.+]] = stablehlo.and
  // CHECK: return %[[AND]]
  return %1 : tensor<2xi32>
}

// CHECK-LABEL: @and_i32_neg_one
//  CHECK-SAME:  (%[[ARG0:.+]]: tensor<2xi32>)
func.func @and_i32_neg_one(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.constant dense<-1> : tensor<2xi32>
  %1 = stablehlo.and %0, %arg0 : tensor<2xi32>
  // CHECK-NOT:  stablehlo.and
  // CHECK: return %[[ARG0]]
  return %1 : tensor<2xi32>
}

// -----

/////////
// BroadcastInDim

// CHECK-LABEL: func.func @broadcast_in_dim_transpose
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<3x3xi32>)
func.func @broadcast_in_dim_transpose(%arg0: tensor<3x3xi32>)
  -> (tensor<3x3xi32>, tensor<3x3xi32>) {
  %3 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<3x3xi32>) -> tensor<3x3xi32>
  %4 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<3x3xi32>) -> tensor<3x3xi32>

  // CHECK: [[R4:%.+]] = stablehlo.transpose [[ARG0]], dims = [1, 0] : (tensor<3x3xi32>) -> tensor<3x3xi32>

  // CHECK-NEXT: return [[ARG0]], [[R4]]
  return %3, %4 : tensor<3x3xi32>, tensor<3x3xi32>
}

// CHECK-LABEL: func.func @broadcast_in_dim_transpose_invert
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x2x3x4xf32>)
func.func @broadcast_in_dim_transpose_invert(%arg0 : tensor<1x2x3x4xf32>) -> tensor<3x1x4x2xf32> {
  // stablehlo.transpose %arg0, dims = [2, 0, 3, 1] : (tensor<1x2x3x4xf32>) -> tensor<3x1x4x2xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1,3,0,2] : (tensor<1x2x3x4xf32>) -> tensor<3x1x4x2xf32>
  return %0 : tensor<3x1x4x2xf32>
}

// CHECK-LABEL: func.func @broadcast_in_dim_nested
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<3x3xi32>)
func.func @broadcast_in_dim_nested(%arg0: tensor<3x3xi32>)
  -> (tensor<3x2x3x3xi32>) {
  %6 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<3x3xi32>) -> tensor<3x3x2xi32>
  %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2, 1] : (tensor<3x3x2xi32>) -> tensor<3x2x3x3xi32>
  // CHECK: [[R6:%.+]] = stablehlo.broadcast_in_dim [[ARG0]], dims = [2, 0] : (tensor<3x3xi32>) -> tensor<3x2x3x3xi32>

  // CHECK-NEXT: return [[R6]]
  return %7 : tensor<3x2x3x3xi32>
}

// CHECK-LABEL: func.func @broadcast_in_dim_nested_bounded
func.func @broadcast_in_dim_nested_bounded(%arg0: tensor<3x3xi32>, %arg1: tensor<i32>) -> tensor<3x2x?x3xi32, #stablehlo.bounds<?, ?, 3, ?>> {
  // CHECK: [[SDS:%.+]] = stablehlo.set_dimension_size
  // CHECK-NEXT: stablehlo.broadcast_in_dim [[SDS]], dims = [2, 0] : (tensor<?x3xi32, #stablehlo.bounds<3, ?>>) -> tensor<3x2x?x3xi32, #stablehlo.bounds<?, ?, 3, ?>>
  %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<3x3xi32>, tensor<i32>) -> tensor<?x3xi32, #stablehlo.bounds<3, ?>>
  %1 = stablehlo.broadcast_in_dim %0, dims = [1, 0] : (tensor<?x3xi32, #stablehlo.bounds<3, ?>>) -> tensor<3x?x2xi32, #stablehlo.bounds<?, 3, ?>>
  %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2, 1] : (tensor<3x?x2xi32, #stablehlo.bounds<?, 3, ?>>) -> tensor<3x2x?x3xi32, #stablehlo.bounds<?, ?, 3, ?>>
  return %2 : tensor<3x2x?x3xi32, #stablehlo.bounds<?, ?, 3, ?>>
}

// CHECK-LABEL: func.func @broadcast_in_dim_reshape
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<3x6xi32>)
func.func @broadcast_in_dim_reshape(%arg0: tensor<3x6xi32>)
  -> (tensor<1x3x6xi32>, tensor<3x6x1xi32>) {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<3x6xi32>) -> tensor<1x3x6xi32>
  %5 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<3x6xi32>) -> tensor<3x6x1xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.reshape [[ARG0]] : (tensor<3x6xi32>) -> tensor<1x3x6xi32>
  // CHECK-DAG:  [[R5:%.+]] = stablehlo.reshape [[ARG0]] : (tensor<3x6xi32>) -> tensor<3x6x1xi32>

  // CHECK-NEXT: return [[R0]], [[R5]]
  return %0, %5 : tensor<1x3x6xi32>, tensor<3x6x1xi32>
}

// CHECK-LABEL: func.func @broadcast_in_dim_bounded_no_reshape
func.func @broadcast_in_dim_bounded_no_reshape(%arg0: tensor<20xf32>, %arg1: tensor<i32>) -> tensor<1x?xf32, #stablehlo.bounds<?, 20>> {
  %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<20xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<20>>
  // CHECK: stablehlo.set_dimension_size
  // CHECK-NEXT: stablehlo.broadcast_in_dim
  %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<?xf32, #stablehlo.bounds<20>>) -> tensor<1x?xf32, #stablehlo.bounds<?, 20>>
  return %1 : tensor<1x?xf32, #stablehlo.bounds<?, 20>>
}

// CHECK-LABEL: func.func @broadcast_in_dim_prefer_nested_reshape
// CHECK-SAME:   ([[ARG0:%[^ ]+]]: tensor<3x4xi32>)
func.func @broadcast_in_dim_prefer_nested_reshape(%arg0: tensor<3x4xi32>) -> (tensor<2x3x4x3xi32>, tensor<2x3x4x3xi32>) {
  // When `broadcast_in_dim(broadcast_in_dim(x))` could be optimized into either
  // `broadcast_in_dim(reshape(x))` or `broadcast_in_dim(x)`, we want to select
  // the former pattern.
  //
  // (We accomplish this by blocking the merge-composition pattern if the inner
  // op can be replaced with a `reshape`. Simply adding benefit to the
  // replace-with-reshape pattern isn't sufficient here because the outermost
  // op, which only matches the merge-composition pattern, is traversed first.)

  // CHECK-DAG: [[INNER_RESHAPE:%[^ ]+]] = stablehlo.reshape [[ARG0]] : (tensor<3x4xi32>) -> tensor<3x1x4xi32>
  // CHECK-DAG: [[BROADCAST_OF_RESHAPE:%[^ ]+]] = stablehlo.broadcast_in_dim [[INNER_RESHAPE]], dims = [1, 0, 2] : (tensor<3x1x4xi32>) -> tensor<2x3x4x3xi32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<3x4xi32>) -> tensor<3x1x4xi32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [1, 0, 2] : (tensor<3x1x4xi32>) -> tensor<2x3x4x3xi32>

  // When the inner op doesn't qualify for replacement with a `reshape` op,
  // however (particularly when it meets some conditions but not others), ensure
  // that we allow the merge-composition pattern to match.

  // CHECK-DAG: [[MERGED_BROADCAST:%[^ ]+]] = stablehlo.broadcast_in_dim [[ARG0]], dims = [3, 2] : (tensor<3x4xi32>) -> tensor<2x3x4x3xi32>
  %2 = stablehlo.broadcast_in_dim %arg0, dims = [2, 1] : (tensor<3x4xi32>) -> tensor<1x4x3xi32>
  %3 = stablehlo.broadcast_in_dim %2, dims = [0, 2, 3] : (tensor<1x4x3xi32>) -> tensor<2x3x4x3xi32>

  // CHECK-DAG: return [[BROADCAST_OF_RESHAPE]], [[MERGED_BROADCAST]]
  return %1, %3 : tensor<2x3x4x3xi32>, tensor<2x3x4x3xi32>
}

// CHECK-LABEL: func.func @broadcast_in_dim_not_identity_broadcasts
func.func @broadcast_in_dim_not_identity_broadcasts(%arg0: tensor<1x2xf32>) -> tensor<2x2xf32> {
  // CHECK: stablehlo.broadcast_in_dim
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

/////////
// CompareOp

// CHECK-LABEL: func.func @compare_signed_arg
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<i32>)
func.func @compare_signed_arg(%arg0: tensor<i32>)
  -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c5 = stablehlo.constant dense<5> : tensor<i32>

  %0 = stablehlo.compare EQ, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = stablehlo.compare GT, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare LE, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare NE, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %4 = stablehlo.compare EQ, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %5 = stablehlo.compare LT, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %6 = stablehlo.compare GE, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %7 = stablehlo.compare NE, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  [[C1:%.+]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK-DAG:  [[C5:%.+]] = stablehlo.constant dense<5> : tensor<i32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.compare EQ, [[ARG0]], [[C5]], SIGNED
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.compare GT, [[ARG0]], [[C5]], SIGNED
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.compare LE, [[ARG0]], [[C5]], SIGNED
  // CHECK-DAG:  [[R3:%.+]] = stablehlo.compare NE, [[ARG0]], [[C5]], SIGNED

  // CHECK-NEXT: return [[C1]], [[C0]], [[C1]], [[C0]], [[R0]], [[R1]], [[R2]], [[R3]]
  return %0, %1, %2, %3, %4, %5, %6, %7 :
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// CHECK-LABEL: func.func @compare_unsigned_arg
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<i32>)
func.func @compare_unsigned_arg(%arg0: tensor<i32>)
  -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c5 = stablehlo.constant dense<5> : tensor<i32>

  %0 = stablehlo.compare EQ, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = stablehlo.compare GT, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare LE, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare NE, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %4 = stablehlo.compare EQ, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %5 = stablehlo.compare LT, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %6 = stablehlo.compare GE, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %7 = stablehlo.compare NE, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  [[C1:%.+]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK-DAG:  [[C5:%.+]] = stablehlo.constant dense<5> : tensor<i32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.compare EQ, [[ARG0]], [[C5]], UNSIGNED
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.compare GT, [[ARG0]], [[C5]], UNSIGNED
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.compare LE, [[ARG0]], [[C5]], UNSIGNED
  // CHECK-DAG:  [[R3:%.+]] = stablehlo.compare NE, [[ARG0]], [[C5]], UNSIGNED

  // CHECK-NEXT: return [[C1]], [[C0]], [[C1]], [[C0]], [[R0]], [[R1]], [[R2]], [[R3]]
  return %0, %1, %2, %3, %4, %5, %6, %7 :
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// CHECK-LABEL: func.func @compare_op_bool_simplify
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<i1>)
func.func @compare_op_bool_simplify(%arg0: tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  %false = stablehlo.constant dense<false> : tensor<i1>
  %true = stablehlo.constant dense<true> : tensor<i1>
  // CHECK-NOT: stablehlo.compare
  %0 = stablehlo.compare NE, %arg0, %false, UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %1 = stablehlo.compare EQ, %arg0, %true, UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
  // CHECK: return [[ARG0]], [[ARG0]]
  func.return %0, %1 : tensor<i1>, tensor<i1>
}

// -----

/////////
// ComplexOp

// CHECK-LABEL: @complex_collapse_simplify
func.func @complex_collapse_simplify(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = stablehlo.real %arg0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %1 = stablehlo.imag %arg0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %2 = stablehlo.complex %0, %1 : tensor<4xcomplex<f32>>
  // CHECK: return %arg0
  return %2 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: @complex_expand_simplify
func.func @complex_expand_simplify(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = stablehlo.complex %arg0, %arg1 : tensor<4xcomplex<f32>>
  %1 = stablehlo.real %0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %2 = stablehlo.imag %0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  // CHECK: return %arg0, %arg1
  return %1, %2 : tensor<4xf32>, tensor<4xf32>
}

// -----

////////
// ConcatenateOp

// CHECK-LABEL: concatenate_noop
func.func @concatenate_noop(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-SAME: [[ARG:%.+]]: tensor<4xi32>
  %0 = "stablehlo.concatenate"(%arg0) <{ dimension = 0 : i64 }> : (tensor<4xi32>) -> tensor<4xi32>

  // CHECK: return [[ARG]]
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_with_empty
func.func @concatenate_with_empty(%arg0: tensor<4xi32>, %arg1: tensor<0xi32>) -> tensor<8xi32> {
  // CHECK-SAME: [[ARG0:%.+]]: tensor<4xi32>
  // CHECK-SAME: [[ARG1:%.+]]: tensor<0xi32>
  // CHECK: stablehlo.concatenate [[ARG0]], [[ARG0]], dim = 0
  %0 = "stablehlo.concatenate"(%arg0, %arg0, %arg1) <{ dimension = 0 : i64 }> : (tensor<4xi32>, tensor<4xi32>, tensor<0xi32>) -> tensor<8xi32>
  func.return %0 : tensor<8xi32>
}


// CHECK-LABEL: concatenate_empty_bool
func.func @concatenate_empty_bool(%arg0: tensor<0xi1>, %arg1: tensor<0xi1>) -> tensor<0xi1> {
  // CHECK: stablehlo.constant dense<>
  %0 = "stablehlo.concatenate"(%arg0, %arg1) <{ dimension = 0 : i64 }> : (tensor<0xi1>, tensor<0xi1>) -> tensor<0xi1>
  func.return %0 : tensor<0xi1>
}

// CHECK-LABEL: concatenate_forward
func.func @concatenate_forward(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<12xi32> {
  // CHECK: [[CST:%.+]] = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  // CHECK: stablehlo.concatenate %arg0, %arg1, [[CST]], dim = 0 : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<12xi32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<4xi32>, tensor<4xi32>) -> tensor<8xi32>
  %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %1 = stablehlo.concatenate %0, %c, dim = 0 : (tensor<8xi32>, tensor<4xi32>) -> tensor<12xi32>
  func.return %1 : tensor<12xi32>
}

// CHECK-LABEL: concatenate_zero_extent
func.func @concatenate_zero_extent(%arg0: tensor<0xi32>, %arg1: tensor<0xi32>) -> tensor<0xi32> {
  // CHECK: stablehlo.constant dense<>
  %0 = "stablehlo.concatenate"(%arg0, %arg1) <{ dimension = 0 : i64 }> : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi32>

  func.return %0 : tensor<0xi32>
}

// CHECK-LABEL: concatenate_empty_float
func.func @concatenate_empty_float(%arg0: tensor<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
  // CHECK: stablehlo.constant dense<>
  %0 = "stablehlo.concatenate"(%arg0, %arg1) <{ dimension = 0 : i64 }> : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>

  func.return %0 : tensor<0xf32>
}

// -----

/////////
// ConvertOp

// CHECK-LABEL: func.func @convert
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xf32>)
func.func @convert(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %r = stablehlo.convert %arg0 : tensor<2xf32>

  // CHECK: return [[ARG0]]
  return %r : tensor<2xf32>
}

// -----

/////////
// CustomCallOp

// CHECK-LABEL: @custom_call_unregistered_backend_config_to_ffi
func.func @custom_call_unregistered_backend_config_to_ffi(%arg0: tensor<1xf32>) -> (tensor<1xf32>) {
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @foo(%arg0) {api_version = 4 : i32, backend_config = {bar = 1 : i32}} : (tensor<1xf32>) -> tensor<1xf32>
  %0 = stablehlo.custom_call @foo(%arg0) {api_version = 1 : i32, mhlo.backend_config = {bar = 1: i32}} : (tensor<1xf32>) -> tensor<1xf32>
  // CHECK-NEXT: return %[[CC]]
  return %0 : tensor<1xf32>
}

// -----

/////////
// DynamicBroadcastInDimOp

// CHECK-LABEL: func @dynamic_broadcast_in_dim_all_dims_non_expanding
func.func @dynamic_broadcast_in_dim_all_dims_non_expanding(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  // CHECK-NEXT: return %[[ARG]]
  %1 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {
    broadcast_dimensions = array<i64: 0>,
    known_expanding_dimensions = array<i64>,
    known_nonexpanding_dimensions = array<i64: 0>
  } : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// CHECK-LABEL: @dynamic_broadcast_of_dynamic_reshape_same_shape
func.func @dynamic_broadcast_of_dynamic_reshape_same_shape(%arg0: tensor<?xf32>, %arg1: tensor<2xi64>) -> tensor<?x?xf32> {
  %0 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  %1 = stablehlo.dynamic_broadcast_in_dim %0, %arg1, dims = [0, 1] : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>

  // CHECK-NOT: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.dynamic_reshape %arg0, %arg1
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xf32>) -> tensor<5x4xf32>
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  func.return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_shape
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_shape(%arg0: tensor<i32>) -> tensor<4x32xi32> {
  %0 = stablehlo.constant dense<[4, 32]> : tensor<2xi32>
  // CHECK: %[[RESULT:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<i32>) -> tensor<4x32xi32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [] : (tensor<i32>, tensor<2xi32>) -> tensor<?x32xi32>
  %2 = stablehlo.dynamic_reshape %1, %0 : (tensor<?x32xi32>, tensor<2xi32>) -> tensor<4x32xi32>
  // CHECK: return %[[RESULT]] : tensor<4x32xi32>
  func.return %2 : tensor<4x32xi32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_index_shape
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_index_shape(%arg0: tensor<f32>) -> tensor<4x32xf32> {
  %0 = shape.const_shape [4, 32] : tensor<2xindex>
  // CHECK: %[[RESULT:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<4x32xf32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
  %2 = stablehlo.dynamic_reshape %1, %0 : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<4x32xf32>
  // CHECK: return %[[RESULT]] : tensor<4x32xf32>
  func.return %2 : tensor<4x32xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_requires_cast
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_requires_cast(%arg0: tensor<f32>) -> tensor<?x?xf32> {
  %0 = shape.const_shape [4, 32] : tensor<2xindex>
  // CHECK: %[[BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<4x32xf32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK: %[[RESULT:.*]] = stablehlo.convert %[[BCAST]] : (tensor<4x32xf32>) -> tensor<?x?xf32>
  // CHECK: return %[[RESULT]] : tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_almost_not_actually_dynamic
func.func @dynamic_broadcast_in_dim_op_almost_not_actually_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  func.return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_shape_of
func.func @dynamic_broadcast_in_dim_to_shape_of(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %2 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %0) <{ broadcast_dimensions = array<i64: 0> }> : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: return %[[ARG]] : tensor<?xf32>
  func.return %2 : tensor<?xf32>
}

// CHECK-LABEL: @dynamic_broadcast_of_reshape
func.func @dynamic_broadcast_of_reshape(%arg: tensor<?xf32>,
                                        %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK: [[RESHAPE:%.*]] = stablehlo.dynamic_reshape
  // CHECK: return [[RESHAPE]]
  %0 = "stablehlo.dynamic_reshape"(%arg, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = "stablehlo.dynamic_broadcast_in_dim"(%0, %shape) { broadcast_dimensions = array<i64: 0, 1> } : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: @dynamic_broadcast_in_dim_of_reshape_permuted
func.func @dynamic_broadcast_in_dim_of_reshape_permuted(%arg: tensor<?xf32>,
    %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK: stablehlo.dynamic_reshape
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  %0 = "stablehlo.dynamic_reshape"(%arg, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = "stablehlo.dynamic_broadcast_in_dim"(%0, %shape) { broadcast_dimensions = array<i64: 1, 0> } : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// -----

/////////
// DynamicGatherOp

// CHECK-LABEL: @simplify_dynamic_gather_i64
func.func @simplify_dynamic_gather_i64(%arg0: tensor<375682x256xf16>, %arg1: tensor<16x64xi64>) -> tensor<16x64x256xf16> {
  %c = stablehlo.constant dense<[1, 256]> : tensor<2xi64>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false}> : (tensor<375682x256xf16>, tensor<16x64xi64>, tensor<2xi64>) -> tensor<16x64x256xf16>
  // CHECK: %[[RET:.+]] = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256>}> : (tensor<375682x256xf16>, tensor<16x64xi64>) -> tensor<16x64x256xf16>
  // CHECK: return %[[RET]]
  return %0 : tensor<16x64x256xf16>
}

// CHECK-LABEL: @simplify_dynamic_gather_i32
func.func @simplify_dynamic_gather_i32(%arg0: tensor<375682x256xf16>, %arg1: tensor<16x64xi64>) -> tensor<16x64x256xf16> {
  %c = stablehlo.constant dense<[1, 256]> : tensor<2xi32>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false}> : (tensor<375682x256xf16>, tensor<16x64xi64>, tensor<2xi32>) -> tensor<16x64x256xf16>
  // CHECK: %[[RET:.+]] = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256>}> : (tensor<375682x256xf16>, tensor<16x64xi64>) -> tensor<16x64x256xf16>
  // CHECK: return %[[RET]]
  return %0 : tensor<16x64x256xf16>
}

// -----

/////////
// DynamicIotaOp

// CHECK-LABEL: @dynamic_iota_broadcast_dim0_index
func.func @dynamic_iota_broadcast_dim0_index(%arg0 : tensor<2xindex>) -> tensor<5x?xi32> {
  // CHECK: [[IOTA:%.+]] = stablehlo.iota dim = 0 : tensor<5xi32>
  // CHECK: [[BROADCAST:%.+]] = stablehlo.dynamic_broadcast_in_dim [[IOTA]], %arg0, dims = [0] : (tensor<5xi32>, tensor<2xindex>) -> tensor<5x?xi32>
  %0 = "stablehlo.dynamic_iota"(%arg0) <{iota_dimension = 0 : i64}> : (tensor<2xindex>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_broadcast_dim0_i32
func.func @dynamic_iota_broadcast_dim0_i32(%arg0 : tensor<2xi32>) -> tensor<5x?xi32> {
  // CHECK: [[IOTA:%.+]] = stablehlo.iota dim = 0 : tensor<5xi32>
  // CHECK: [[BROADCAST:%.+]] = stablehlo.dynamic_broadcast_in_dim [[IOTA]], %arg0, dims = [0] : (tensor<5xi32>, tensor<2xi32>) -> tensor<5x?xi32>
  %0 = "stablehlo.dynamic_iota"(%arg0) <{iota_dimension = 0 : i64}> : (tensor<2xi32>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_broadcast_dim0_i64
func.func @dynamic_iota_broadcast_dim0_i64(%arg0 : tensor<2xi64>) -> tensor<5x?xi32> {
  // CHECK: [[IOTA:%.+]] = stablehlo.iota dim = 0 : tensor<5xi32>
  // CHECK: [[BROADCAST:%.+]] = stablehlo.dynamic_broadcast_in_dim [[IOTA]], %arg0, dims = [0] : (tensor<5xi32>, tensor<2xi64>) -> tensor<5x?xi32>
  %0 = "stablehlo.dynamic_iota"(%arg0) <{iota_dimension = 0 : i64}> : (tensor<2xi64>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_broadcast_dim1_index
func.func @dynamic_iota_broadcast_dim1_index(%arg0 : tensor<2xindex>) -> tensor<5x?xi32> {
  // CHECK-NEXT: [[CAST:%.+]] = arith.index_cast %arg0 : tensor<2xindex> to tensor<2xi64>
  // CHECK-NEXT: [[SLICE:%.+]] = stablehlo.slice [[CAST]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK-NEXT: [[IOTA:%.+]] = stablehlo.dynamic_iota [[SLICE]], dim = 0 : (tensor<1xi64>) -> tensor<?xi32>
  // CHECK-NEXT: [[BROADCAST:%.+]] = stablehlo.dynamic_broadcast_in_dim [[IOTA]], %arg0, dims = [1] : (tensor<?xi32>, tensor<2xindex>) -> tensor<5x?xi32>
  %0 = "stablehlo.dynamic_iota"(%arg0) <{iota_dimension = 1 : i64}> : (tensor<2xindex>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_broadcast_dim1_i32
func.func @dynamic_iota_broadcast_dim1_i32(%arg0 : tensor<2xi32>) -> tensor<5x?xi32> {
  // CHECK-NEXT: [[CAST:%.+]] = stablehlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xi64>
  // CHECK-NEXT: [[SLICE:%.+]] = stablehlo.slice [[CAST]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK-NEXT: [[IOTA:%.+]] = stablehlo.dynamic_iota [[SLICE]], dim = 0 : (tensor<1xi64>) -> tensor<?xi32>
  // CHECK-NEXT: [[BROADCAST:%.+]] = stablehlo.dynamic_broadcast_in_dim [[IOTA]], %arg0, dims = [1] : (tensor<?xi32>, tensor<2xi32>) -> tensor<5x?xi32>
  %0 = "stablehlo.dynamic_iota"(%arg0) <{iota_dimension = 1 : i64}> : (tensor<2xi32>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_broadcast_dim1_i64
func.func @dynamic_iota_broadcast_dim1_i64(%arg0 : tensor<2xi64>) -> tensor<5x?xi32> {
  // CHECK-NEXT: [[SLICE:%.+]] = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK-NEXT: [[IOTA:%.+]] = stablehlo.dynamic_iota [[SLICE]], dim = 0 : (tensor<1xi64>) -> tensor<?xi32>
  // CHECK-NEXT: [[BROADCAST:%.+]] = stablehlo.dynamic_broadcast_in_dim [[IOTA]], %arg0, dims = [1] : (tensor<?xi32>, tensor<2xi64>) -> tensor<5x?xi32>
  %0 = "stablehlo.dynamic_iota"(%arg0) <{iota_dimension = 1 : i64}> : (tensor<2xi64>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_is_static
func.func @dynamic_iota_is_static(%arg0 : tensor<1xindex>) -> tensor<4xi32> {
  // CHECK: [[RESULT:%.*]] = stablehlo.iota
  // CHECK: return [[RESULT]]
  %0 = "stablehlo.dynamic_iota"(%arg0) <{iota_dimension = 0 : i64}> : (tensor<1xindex>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

/////////
// DynamicPadOp

// CHECK-LABEL: func.func @dynamic_pad_to_pad
func.func @dynamic_pad_to_pad(%arg0: tensor<2x3xi32>, %arg1: tensor<i32>) -> tensor<5x9xi32> {
  %low = stablehlo.constant dense<[0, 1]> : tensor<2xi32>
  %high = stablehlo.constant dense<[2, 1]> : tensor<2xi32>
  %interior = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
  %0 = stablehlo.dynamic_pad %arg0, %arg1, %low, %high, %interior
         : (tensor<2x3xi32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<5x9xi32>
  // CHECK: [[PAD:%.*]] = stablehlo.pad %arg0, %arg1, low = [0, 1], high = [2, 1], interior = [1, 2] : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
  // CHECK: return [[PAD]]
  func.return %0 : tensor<5x9xi32>
}

// -----

/////////
// DynamicReshapeOp

// CHECK-LABEL: func.func @dynamic_reshape_is_static
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1xf32>, [[ARG1:%.+]]: tensor<?x?xf32>, [[ARG2:%.+]]: tensor<2xi32>)
func.func @dynamic_reshape_is_static(%arg0: tensor<1xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<2xi32>)
          -> (tensor<1x1xf32>, tensor<2x1xf32>, tensor<1x2xi32>) {
  %c0 = stablehlo.constant dense<[2, 1]> : tensor<2xi32>

  %0 = stablehlo.dynamic_reshape %arg0, %arg2 : (tensor<1xf32>, tensor<2xi32>) -> tensor<1x1xf32>
  %1 = stablehlo.dynamic_reshape %arg1, %c0 : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<2x1xf32>
  %2 = stablehlo.dynamic_reshape %arg2, %arg2 : (tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.reshape [[ARG0]] : (tensor<1xf32>) -> tensor<1x1xf32>
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.reshape [[ARG1]] : (tensor<?x?xf32>) -> tensor<2x1xf32>
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.reshape [[ARG2]] : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK-NEXT: return [[R0]], [[R1]], [[R2]]
  return %0, %1, %2 : tensor<1x1xf32>, tensor<2x1xf32>, tensor<1x2xi32>
}

// CHECK-LABEL: func @dynamic_reshape_shape_of
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func.func @dynamic_reshape_shape_of(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<2xindex> {
  // CHECK: return [[ARG1]]
  %0 = "stablehlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
  func.return %1 : tensor<2xindex>
}

// CHECK-LABEL: func @dynamic_reshape_of_same_operand_result
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func.func @dynamic_reshape_of_same_operand_result(%arg0: tensor<?xf16>, %arg1: tensor<1xindex>) -> tensor<?xf16> {
  %0 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<?xf16>, tensor<1xindex>) -> tensor<?xf16>
  %1 = stablehlo.add %0, %0 : tensor<?xf16>
  %2 = stablehlo.dynamic_reshape %1, %arg1 : (tensor<?xf16>, tensor<1xindex>) -> tensor<?xf16>
  // CHECK: [[ADD:%.+]] = stablehlo.add
  // CHECK: return [[ADD]]
  return %2 : tensor<?xf16>
}

// -----

/////////
// DynamicSliceOp

// CHECK-LABEL: dynamic_slice_variable_start
func.func @dynamic_slice_variable_start(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // CHECK: stablehlo.dynamic_slice
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [1, 4] : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start
func.func @dynamic_slice_constant_start(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  // CHECK: stablehlo.slice %arg0 [1:3] : (tensor<4xi32>) -> tensor<2xi32>
  %c = stablehlo.constant dense<1> : tensor<i64>
  %0 = stablehlo.dynamic_slice %arg0, %c, sizes = [2] : (tensor<4xi32>, tensor<i64>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start_dynamic_shape
func.func @dynamic_slice_constant_start_dynamic_shape(%arg0: tensor<?x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: stablehlo.dynamic_slice
  // CHECK-NOT: stablehlo.slice
  %c = stablehlo.constant dense<1> : tensor<i64>
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.dynamic_slice %arg0, %c, %c_0, sizes = [1, 4] : (tensor<?x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start_upper_bound
func.func @dynamic_slice_constant_start_upper_bound(%arg0: tensor<8x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: stablehlo.slice %arg0 [7:8, 0:4] : (tensor<8x4xi32>) -> tensor<1x4xi32>
  %c = stablehlo.constant dense<10> : tensor<i64>
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.dynamic_slice %arg0, %c, %c_0, sizes = [1, 4] : (tensor<8x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start_lower_bound
func.func @dynamic_slice_constant_start_lower_bound(%arg0: tensor<8x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: stablehlo.slice %arg0 [0:1, 0:4] : (tensor<8x4xi32>) -> tensor<1x4xi32>
  %c = stablehlo.constant dense<-1> : tensor<i64>
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.dynamic_slice %arg0, %c, %c_0, sizes = [1, 4] : (tensor<8x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

////////
// DynamicSliceOp

// CHECK-LABEL: dynamic_update_slice_noop
func.func @dynamic_update_slice_noop(%arg0: tensor<3x4xi64>, %arg1: tensor<3x0xi64>) -> tensor<3x4xi64> {
  // CHECK: return %arg0
  %c = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c, %c : (tensor<3x4xi64>, tensor<3x0xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// CHECK-LABEL: dynamic_update_slice_noop_dynamic
func.func @dynamic_update_slice_noop_dynamic(%arg0: tensor<?x?xi64>, %arg1: tensor<?x?xi64>) -> tensor<?x?xi64> {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c, %c : (tensor<?x?xi64>, tensor<?x?xi64>, tensor<i64>, tensor<i64>) -> tensor<?x?xi64>
  func.return %0 : tensor<?x?xi64>
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[VAL:.*]] = stablehlo.dynamic_update_slice %arg0, %arg1, %[[CST]], %[[CST]] : (tensor<?x?xi64>, tensor<?x?xi64>, tensor<i64>, tensor<i64>) -> tensor<?x?xi64>
  // CHECK: return %[[VAL]] : tensor<?x?xi64>
}

// CHECK-LABEL: dynamic_update_slice_identity_update
func.func @dynamic_update_slice_identity_update(%arg0: tensor<3x4xi64>, %arg1: tensor<3x4xi64>) -> tensor<3x4xi64> {
  // CHECK: return %arg1
  %c = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c, %c : (tensor<3x4xi64>, tensor<3x4xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

/////////
// GatherOp

// CHECK-LABEL: func.func @gather_to_slice
func.func @gather_to_slice(%arg0: tensor<5x6x7xf32>) -> tensor<3x6x5xf32> {
  %0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      index_vector_dim = 0,
      offset_dims = [0, 1, 2],
      start_index_map = [0, 2],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 3, 6, 5>} : (tensor<5x6x7xf32>, tensor<2xi32>) -> tensor<3x6x5xf32>
  return %1 : tensor<3x6x5xf32>
  // CHECK:      %[[RET:.*]] = stablehlo.slice %arg0 [1:4, 0:6, 2:7]
  // CHECK-SAME:    : (tensor<5x6x7xf32>) -> tensor<3x6x5xf32>
  // CHECK-NEXT: return %[[RET]] : tensor<3x6x5xf32>
}

// CHECK-LABEL: func.func @gather_scalar_index_to_slice
func.func @gather_scalar_index_to_slice(%arg0: tensor<5x6x7xf32>) -> tensor<5x6x4xf32> {
  %0 = arith.constant dense<1> : tensor<i32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      index_vector_dim = 0,
      offset_dims = [0, 1, 2],
      start_index_map = [2],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 5, 6, 4>} : (tensor<5x6x7xf32>, tensor<i32>) -> tensor<5x6x4xf32>
  return %1 : tensor<5x6x4xf32>
  // CHECK:      %[[RET:.*]] = stablehlo.slice %arg0 [0:5, 0:6, 1:5]
  // CHECK-SAME:    : (tensor<5x6x7xf32>) -> tensor<5x6x4xf32>
  // CHECK-NEXT: return %[[RET]] : tensor<5x6x4xf32>
}

// CHECK-LABEL: func.func @gather_to_slice_reshape
func.func @gather_to_slice_reshape(%arg0: tensor<5x6x7xf32>) -> tensor<3x6xf32> {
  %0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [2],
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0, 2],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 3, 6, 1>} : (tensor<5x6x7xf32>, tensor<2xi32>) -> tensor<3x6xf32>
  return %1 : tensor<3x6xf32>
  // CHECK:      %[[V0:.*]] = stablehlo.slice %arg0 [1:4, 0:6, 2:3]
  // CHECK-SAME:    : (tensor<5x6x7xf32>) -> tensor<3x6x1xf32>
  // CHECK-NEXT: %[[V1:.*]] = stablehlo.reshape %[[V0]] : (tensor<3x6x1xf32>) -> tensor<3x6xf32>
  // CHECK-NEXT: return %[[V1]] : tensor<3x6xf32>
}

// CHECK-LABEL: func.func @gather_to_slice_indices_clamp_upperbound
func.func @gather_to_slice_indices_clamp_upperbound(%arg0 : tensor<4x2xui32>) -> tensor<2xui32> {
  %0 = arith.constant dense<4> : tensor<1xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0],
      index_vector_dim = 0,
      collapsed_slice_dims = [0],
      start_index_map = [0]
    >, indices_are_sorted = true,
    slice_sizes = array<i64: 1, 2>} : (tensor<4x2xui32>, tensor<1xi32>) -> tensor<2xui32>
  return %1 : tensor<2xui32>
  // CHECK:      %[[V0:.*]] = stablehlo.slice %arg0 [3:4, 0:2]
  // CHECK-SAME:    : (tensor<4x2xui32>) -> tensor<1x2xui32>
  // CHECK-NEXT: %[[V1:.*]] = stablehlo.reshape %[[V0]] : (tensor<1x2xui32>) -> tensor<2xui32>
  // CHECK-NEXT: return %[[V1]] : tensor<2xui32>
}

// CHECK-LABEL: func.func @gather_to_slice_indices_clamp_lowerbound
func.func @gather_to_slice_indices_clamp_lowerbound(%arg0 : tensor<4x2xui32>) -> tensor<2xui32> {
  %0 = arith.constant dense<-1> : tensor<1xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0],
      index_vector_dim = 0,
      collapsed_slice_dims = [0],
      start_index_map = [0]
    >, indices_are_sorted = true,
    slice_sizes = array<i64: 1, 2>} : (tensor<4x2xui32>, tensor<1xi32>) -> tensor<2xui32>
  return %1 : tensor<2xui32>
  // CHECK:      %[[V0:.*]] = stablehlo.slice %arg0 [0:1, 0:2]
  // CHECK-SAME:    : (tensor<4x2xui32>) -> tensor<1x2xui32>
  // CHECK-NEXT: %[[V1:.*]] = stablehlo.reshape %[[V0]] : (tensor<1x2xui32>) -> tensor<2xui32>
  // CHECK-NEXT: return %[[V1]] : tensor<2xui32>
}

// -----

/////////
// GetDimensionSizeOp

// CHECK-LABEL: func.func @get_dimension_size
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x2x3xf32>, [[ARG1:%.+]]: tensor<?x2xf32>)
func.func @get_dimension_size(%arg0: tensor<1x2x3xf32>, %arg1: tensor<?x2xf32>)
          -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
  %a = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<1x2x3xf32>) -> tensor<i32>
  %b = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<1x2x3xf32>) -> tensor<i32>
  %c = stablehlo.get_dimension_size %arg0, dim = 2 : (tensor<1x2x3xf32>) -> tensor<i32>

  %d = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x2xf32>) -> tensor<i32>
  %e = stablehlo.get_dimension_size %arg1, dim = 1 : (tensor<?x2xf32>) -> tensor<i32>

  // CHECK-DAG:  [[CST1:%.+]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-DAG:  [[CST2:%.+]] = stablehlo.constant dense<2> : tensor<i32>
  // CHECK-DAG:  [[CST3:%.+]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK-DAG:  [[DYN:%.+]]  = stablehlo.get_dimension_size [[ARG1]], dim = 0 : (tensor<?x2xf32>) -> tensor<i32>
  // CHECK-NEXT: return [[CST1]], [[CST2]], [[CST3]], [[DYN]], [[CST2]]
  return %a, %b, %c, %d, %e : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
}

// -----

/////////
// GetTupleElementOp

// CHECK-LABEL: func.func @get_tuple_element
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<f32>, [[ARG1:%.+]]: tensor<i32>, [[ARG2:%.+]]: tuple<tensor<f32>, tensor<f16>>)
func.func @get_tuple_element(%arg0: tensor<f32>, %arg1: tensor<i32>, %arg2: tuple<tensor<f32>, tensor<f16>>)
          -> (tensor<f32>, tensor<i32>, tensor<f16>) {
  %t = stablehlo.tuple %arg0, %arg1 : tuple<tensor<f32>, tensor<i32>>

  %a = stablehlo.get_tuple_element %t[0] : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  %b = stablehlo.get_tuple_element %t[1] : (tuple<tensor<f32>, tensor<i32>>) -> tensor<i32>

  %c = stablehlo.get_tuple_element %arg2[1] : (tuple<tensor<f32>, tensor<f16>>) -> tensor<f16>

  // CHECK:      [[GTE:%.+]] = stablehlo.get_tuple_element [[ARG2]][1] : (tuple<tensor<f32>, tensor<f16>>) -> tensor<f16>
  // CHECK-NEXT: return [[ARG0]], [[ARG1]], [[GTE]]
  return %a, %b, %c : tensor<f32>, tensor<i32>, tensor<f16>
}

// -----

/////////
// ImagOp / RealOp

// CHECK-LABEL: func.func @complex
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xf32>, [[ARG1:%.+]]: tensor<2xf32>)
func.func @complex(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %c = stablehlo.complex %arg0, %arg1 : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  %r = stablehlo.real %c : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %i = stablehlo.imag %c : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[ARG0]], [[ARG1]]
  return %r, %i : tensor<2xf32>, tensor<2xf32>
}

/////////
// IotaOp

// CHECK-LABEL: @iota_constant
func.func @iota_constant() -> tensor<1xi32> {
  // CHECK: [[CONST:%.+]] = stablehlo.constant dense<0> : tensor<1xi32>
  %0 = stablehlo.iota dim = 0 : tensor<1xi32>

  // CHECK: return [[CONST]] : tensor<1xi32>
  func.return %0 : tensor<1xi32>
}

// CHECK-LABEL: @iota_constant_multi
func.func @iota_constant_multi() -> tensor<1x4xi32> {
  // CHECK: [[CONST:%.+]] = stablehlo.constant dense<0> : tensor<1x4xi32>
  %0 = stablehlo.iota dim = 0 : tensor<1x4xi32>

  // CHECK: return [[CONST]] : tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}


// CHECK-LABEL: @iota_not_lowered_to_constant
func.func @iota_not_lowered_to_constant() -> tensor<4xi32> {
  // CHECK: [[RESULT:%.*]] = stablehlo.iota
  // CHECK: return [[RESULT]]
  %0 = stablehlo.iota dim = 0 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @iota_broadcast_dim1
func.func @iota_broadcast_dim1() -> tensor<5x4xi32> {
  // CHECK: [[IOTA:%.+]] = stablehlo.iota dim = 0 : tensor<5xi32>
  // CHECK: [[RESULT:%.+]] = stablehlo.broadcast_in_dim [[IOTA]], dims = [0] : (tensor<5xi32>) -> tensor<5x4xi32>
  %0 = stablehlo.iota dim = 0 : tensor<5x4xi32>

  func.return %0 : tensor<5x4xi32>
}

// CHECK-LABEL: @iota_broadcast_dim2
func.func @iota_broadcast_dim2() -> tensor<5x4xi32> {
  // CHECK: [[IOTA:%.+]] = stablehlo.iota dim = 0 : tensor<4xi32>
  // CHECK: [[RESULT:%.+]] = stablehlo.broadcast_in_dim [[IOTA]], dims = [1] : (tensor<4xi32>) -> tensor<5x4xi32>
  %0 = stablehlo.iota dim = 1 : tensor<5x4xi32>

  func.return %0 : tensor<5x4xi32>
}

// -----

/////////
// MaxOp

// CHECK-LABEL: @maximum_cst_on_rhs
func.func @maximum_cst_on_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<2.0> : tensor<f32>
  // CHECK: stablehlo.maximum %arg0, %cst : tensor<f32>
  %0 = stablehlo.maximum %cst, %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

/////////
// MinOp

// CHECK-LABEL: @minimum_cst_on_rhs
func.func @minimum_cst_on_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<2.0> : tensor<f32>
  // CHECK: stablehlo.minimum %arg0, %cst : tensor<f32>
  %0 = stablehlo.minimum %cst, %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

/////////
// MulOp

// CHECK-LABEL: @multiply_cst_on_rhs
func.func @multiply_cst_on_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<2.0> : tensor<f32>
  // CHECK: stablehlo.multiply %arg0, %cst : tensor<f32>
  %0 = stablehlo.multiply %cst, %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @multiply_by_zero
func.func @multiply_by_zero(%arg0: tensor<i32>) -> tensor<i32> {
  %cst = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.multiply %cst, %arg0 : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: @multiply_by_one
func.func @multiply_by_one(%arg0: tensor<i32>) -> tensor<i32> {
  %cst = stablehlo.constant dense<1> : tensor<i32>
  %0 = stablehlo.multiply %cst, %arg0 : tensor<i32>
  // CHECK-NOT: stablehlo.constant
  // CHECK: return %arg0 : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: @multiply_by_zero_float
func.func @multiply_by_zero_float(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: stablehlo.constant dense<0.0{{.*}}> : tensor<f32>
  %0 = stablehlo.multiply %cst, %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @multiply_by_one_float
func.func @multiply_by_one_float(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<1.0> : tensor<f32>
  %0 = stablehlo.multiply %cst, %arg0 : tensor<f32>
  // CHECK-NOT: stablehlo.constant
  // CHECK: return %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @multiply_by_one_merge_attrs
func.func @multiply_by_one_merge_attrs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<1.0> : tensor<f32>
  %0 = stablehlo.add %arg0, %arg0 {mhlo.frontend_attributes = {bar = "1"}} : tensor<f32>
  %1 = stablehlo.multiply %0, %cst {mhlo.frontend_attributes = {foo = "1"}} : tensor<f32>
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {mhlo.frontend_attributes = {bar = "1", foo = "1"}} : tensor<f32>
  // CHECK: return %[[ADD]] : tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @multiply_by_one_merge_attrs_conflict
func.func @multiply_by_one_merge_attrs_conflict(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<1.0> : tensor<f32>
  %0 = stablehlo.add %arg0, %arg0 {mhlo.frontend_attributes = {bar = "1", foo = "0"}} : tensor<f32>
  %1 = stablehlo.multiply %0, %cst {mhlo.frontend_attributes = {foo = "1"}} : tensor<f32>
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {mhlo.frontend_attributes = {bar = "1", foo = "1"}} : tensor<f32>
  // CHECK: return %[[ADD]] : tensor<f32>
  return %1 : tensor<f32>
}

// -----

/////////
// OrOp

// CHECK-LABEL: @or_cst_on_rhs
func.func @or_cst_on_rhs(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %cst = stablehlo.constant dense<false> : tensor<2xi1>
  %0 = stablehlo.or %cst, %arg0 : tensor<2xi1>
  // Check that constant canonicalized to RHS, then other patterns apply
  // CHECK-NOT: stablehlo.or
  // CHECK: return %arg0
  return %0 : tensor<2xi1>
}

// CHECK-LABEL: @or_zero
func.func @or_zero(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = stablehlo.constant dense<false> : tensor<2xi1>
  %1 = stablehlo.or %0, %arg0 : tensor<2xi1>

  // CHECK-NOT: stablehlo.or
  // CHECK: return %arg0
  return %1 : tensor<2xi1>
}

// CHECK-LABEL: @or_one
func.func @or_one(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = stablehlo.constant dense<true> : tensor<2xi1>
  %1 = stablehlo.or %0, %arg0 : tensor<2xi1>

  // CHECK-NOT: stablehlo.or
  // CHECK: [[TRUE:%.+]] = stablehlo.constant dense<true> : tensor<2xi1>
  // CHECK: return [[TRUE]]
  return %1 : tensor<2xi1>
}

// CHECK-LABEL: @or_i32_one
func.func @or_i32_one(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.constant dense<1> : tensor<2xi32>
  %1 = stablehlo.or %0, %arg0 : tensor<2xi32>
  // CHECK: %[[OR:.+]] = stablehlo.or
  // CHECK: return %[[OR]]
  return %1 : tensor<2xi32>
}

// CHECK-LABEL: @or_i32_neg_one
func.func @or_i32_neg_one(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.constant dense<-1> : tensor<2xi32>
  %1 = stablehlo.or %0, %arg0 : tensor<2xi32>
  // CHECK-NOT: stablehlo.or
  // CHECK: [[NEG_ONE:%.+]] = stablehlo.constant dense<-1> : tensor<2xi32>
  // CHECK: return [[NEG_ONE]]
  return %1 : tensor<2xi32>
}

// -----

////////
// PadOp

// CHECK-LABEL: @pad_zero_length
func.func @pad_zero_length(%arg0: tensor<5x0xf32>, %arg1: tensor<f32>) -> tensor<7x2xf32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [1, 1], high = [1, 1], interior = [0, 0]
    : (tensor<5x0xf32>, tensor<f32>) -> tensor<7x2xf32>
  // CHECK: %[[RES:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<7x2xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<7x2xf32>
}

// Can't do anything with the dynamic shape, but shouldn't crash.
// CHECK-LABEL: @dynamic_pad
func.func @dynamic_pad(%arg0: tensor<?x2x3xi1>, %arg1: tensor<i1>) -> tensor<?x2x1xi1> {
  %0 = stablehlo.pad %arg0, %arg1, low = [0, 0, -1], high = [0, 0, -1], interior = [0, 0, 0] : (tensor<?x2x3xi1>, tensor<i1>) -> tensor<?x2x1xi1>
  // CHECK-NEXT: %[[RES:.+]] = stablehlo.pad %arg0, %arg1, low = [0, 0, -1], high = [0, 0, -1], interior = [0, 0, 0] : (tensor<?x2x3xi1>, tensor<i1>) -> tensor<?x2x1xi1>
  // CHECK-NEXT: return %[[RES]]
  return %0 : tensor<?x2x1xi1>
}

// CHECK-LABEL: @pad_noop
func.func @pad_noop(%arg0: tensor<256x1024xbf16>, %arg1: tensor<i32>) -> tensor<256x1024xbf16> {
  %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<bf16>
  // CHECK-NOT: stablehlo.pad
  %1 = stablehlo.pad %arg0, %0, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<256x1024xbf16>, tensor<bf16>) -> tensor<256x1024xbf16>
  return %1 : tensor<256x1024xbf16>
}

// We don't want to delete `pad` ops that move a tensor's values around without
// affecting its dimensions.
//
// CHECK-LABEL: @pad_rotate_tensor_no_dim_change
func.func @pad_rotate_tensor_no_dim_change(%arg0: tensor<50x50xf32>) -> tensor<50x50xf32> {
  // CHECK: %[[RES:.+]] = stablehlo.pad
  // CHECK: return %[[RES]]
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [0, -1], high = [0, 1], interior = [0, 0] : (tensor<50x50xf32>, tensor<f32>) -> tensor<50x50xf32>
  return %0 : tensor<50x50xf32>
}

// -----

/////////
// RealDynamicSliceOp

// CHECK-LABEL: @simplify_real_dynamic_slice_to_slice
func.func @simplify_real_dynamic_slice_to_slice(%arg0: tensor<?x4xf32>) -> tensor<1x4xf32> {
  %0 = stablehlo.constant dense<[0, 0]> : tensor<2xi32>
  %1 = stablehlo.constant dense<[1, 4]> : tensor<2xi32>
  %2 = stablehlo.constant dense<[1, 1]> : tensor<2xi32>
  %3 = stablehlo.real_dynamic_slice %arg0, %0, %1, %2 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x4xf32>
  // CHECK: %[[RESULT:.*]] =  stablehlo.slice %arg0 [0:1, 0:4] : (tensor<?x4xf32>) -> tensor<1x4xf32>
  // CHECK: return %[[RESULT]] : tensor<1x4xf32>
  return %3 : tensor<1x4xf32>
}

// CHECK-LABEL: @simplify_real_dynamic_slice_to_dynamic_slice
func.func @simplify_real_dynamic_slice_to_dynamic_slice(%arg0: tensor<?x4xf32>, %arg1: tensor<2xi32>) -> tensor<1x4xf32> {
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi32>
  %1 = stablehlo.add %arg1, %0 : tensor<2xi32>
  %2 = stablehlo.constant dense<[1, 1]> : tensor<2xi32>
  %3 = stablehlo.real_dynamic_slice %arg0, %arg1, %1, %2 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x4xf32>
  return %3 : tensor<1x4xf32>
  //      CHECK: [[START_INDEX_0_1D:%.*]] = stablehlo.slice %arg1 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: [[START_INDEX_0_0D:%.*]] = stablehlo.reshape [[START_INDEX_0_1D]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: [[START_INDEX_1_1D:%.*]] = stablehlo.slice %arg1 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: [[START_INDEX_1_0D:%.*]] = stablehlo.reshape [[START_INDEX_1_1D]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: [[RESULT:%.*]] = stablehlo.dynamic_slice %arg0, [[START_INDEX_0_0D]], [[START_INDEX_1_0D]], sizes = [1, 4] : (tensor<?x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x4xf32>
}

// -----

/////////
// ReduceOp

// CHECK-LABEL: @reduce_no_dimensions
func.func @reduce_no_dimensions(%arg0: tensor<8xi64>, %arg1: tensor<8xi64>) -> (tensor<8xi64>, tensor<8xi64>) {
  %c = stablehlo.constant dense<1> : tensor<i64>
  %0:2 = stablehlo.reduce(%arg0 init: %c), (%arg1 init: %c) across dimensions = [] : (tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>) -> (tensor<8xi64>, tensor<8xi64>)
    reducer(%arg2: tensor<i64>, %arg4: tensor<i64>) (%arg3: tensor<i64>, %arg5: tensor<i64>)  {
    %1 = stablehlo.add %arg2, %arg4 : tensor<i64>
    %2 = stablehlo.subtract %arg3, %arg5 : tensor<i64>
    stablehlo.return %1, %2 : tensor<i64>, tensor<i64>
  }
  // CHECK-NOT: stablehlo.reduce
  // CHECK: return %arg0, %arg1
  return %0#0, %0#1 : tensor<8xi64>, tensor<8xi64>
}

// CHECK-LABEL: func.func @reduce_noop_2
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<4x8xi32>, [[ARG1:%.+]]: tensor<i32>)
func.func @reduce_noop_2(%arg0: tensor<4x8xi32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.reduce(%arg0 init: %0) across dimensions = [0, 1] : (tensor<4x8xi32>, tensor<i32>) -> tensor<i32>
    reducer(%b1: tensor<i32>, %b2: tensor<i32>) {
    stablehlo.return %arg1 : tensor<i32>
  }
  // CHECK: return [[ARG1]] : tensor<i32>
  func.return %1 : tensor<i32>
}


// Each reduce_unused_* test case is accompanied by an ASCII diagram that
// represents the surveyed reduce operation in a compact form:
//
//  (a, a1) (b, b1)
//    ▴       ▴
//    │       │
//    │       │
//    │       │
//    │       │
//    r0      r1
//    U
//
// In this case:
// a,  b  - are the operands to be reduced, i.e. %arg0, %arg1.
// a1, b1 - are the initial values of the operands, i.e. %0, %1.
// r0, r1 - are the results of stablehlo.reduce and/or stablehlo.return operation
//          (they are equivalent in this context), i.e %2#0, %2#1 and %3, %4.
// Arrows show which results depend on which inputs. More specifically it
// represents a set of instructions followed along the def-use chain starting
// from the return operand and a set of reachable operand pairs that conclude
// these chains:
// r0 is an alias for %3, set of instructions { %3 }
//                        set of reachable operand pairs { (%arg2, %arg3) }
// r1 is an alias for %4, set of instructions { %4 }
//                        set of reachable operand pairs { (%arg4, %arg5) }
// U below the result means it is used (live).
//
// To drop r1 use pair (b, b1).

// CHECK-LABEL: func.func @reduce_unused_case0
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case0(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.reduce([[ARG0]] init: [[R0]]) applies stablehlo.add
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>)
   reducer(%arg2: tensor<i64>, %arg3: tensor<i64>)
          (%arg4: tensor<i64>, %arg5: tensor<i64>)
  {
    %3 = stablehlo.add %arg2, %arg3 : tensor<i64>
    %4 = stablehlo.subtract %arg4, %arg5 : tensor<i64>
    stablehlo.return %3, %4 : tensor<i64>, tensor<i64>
  }
  return %2#0 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       │       │
//    │       │       │
//    │       │       │
//    │       │       │
//    r0      r1      r2
//    U               U
//
// To drop r1 use (b, b1).

// CHECK-LABEL: func.func @reduce_unused_case1
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case1(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> (tensor<i64>, tensor<i64>) {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:2 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.minimum %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.maximum %arg5, %arg8 : tensor<i64>
    stablehlo.return %4, %5, %6 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0, %3#2 : tensor<i64>, tensor<i64>
}
// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴
//    │       │
//    │       │
//    │       │◄──────┐
//    │       │       │
//    r0      r1      r2
//    U       U       U
//
// All results are used.

// CHECK-LABEL: func.func @reduce_unused_case2
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case2(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> (tensor<i64>, tensor<i64>, tensor<i64>) {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.minimum %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.maximum %arg4, %arg7 : tensor<i64>
    stablehlo.return %4, %5, %6 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0, %3#1, %3#2 : tensor<i64>, tensor<i64>, tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴
//    │       │
//    │       │
//    │       │◄──────┐
//    │       │       │
//    r0      r1      r2
//                    U
//
// To drop r0 and r1 use pairs (a, a1) (c, c1).

// CHECK-LABEL: func.func @reduce_unused_case3
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case3(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R1:%.+]] = stablehlo.reduce([[ARG1]] init: [[R0]]) applies stablehlo.maximum
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.minimum %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.maximum %arg4, %arg7 : tensor<i64>
    stablehlo.return %4, %5, %6 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#2 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       │       │
//    ├───────┼───────┘
//    │       │◄──────┐
//    │       │       │
//    r0      r1      r2
//    U               U
//
// There is no suitable pair to drop r1.

// CHECK-LABEL: func.func @reduce_unused_case4
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case4(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> (tensor<i64>, tensor<i64>) {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.subtract %4, %arg8 : tensor<i64>
    %6 = stablehlo.minimum %arg4, %arg7 : tensor<i64>
    %7 = stablehlo.maximum %arg4, %arg7 : tensor<i64>
    stablehlo.return %5, %6, %7 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0, %3#2 : tensor<i64>, tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       │       │
//    ├───────┼───────┘
//    │       │◄──────┐
//    │       │       │
//    r0      r1      r2
//    U
//
// There is 1 suitable pair (b, b1), but used by 2 return operands.

// CHECK-LABEL: func.func @reduce_unused_case5
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case5(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.subtract %4, %arg8 : tensor<i64>
    %6 = stablehlo.minimum %arg4, %arg7 : tensor<i64>
    %7 = stablehlo.maximum %arg4, %arg7 : tensor<i64>
    stablehlo.return %5, %6, %7 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴               ▴
//    │               │
//    │       ┌───────┘
//    │◄──────┼───────┐
//    │       │       │
//    r0      r1      r2
//    U
//
// Both r1 and r2 can be dropped using (c, c1) & (b, b1) pairs.

// CHECK-LABEL: func.func @reduce_unused_case6
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case6(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.reduce([[ARG0]] init: [[R0]]) applies stablehlo.add
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.minimum %arg3, %arg6 : tensor<i64>
    %6 = stablehlo.maximum %arg5, %arg8 : tensor<i64>
    stablehlo.return %4, %6, %5 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       │       │
//    │       ├──────►│
//    │       │◄──────┤
//    │       │       │
//    r0      r1      r2
//    U
//
// Both r1 and r2 can be dropped.

// CHECK-LABEL: func.func @reduce_unused_case7
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case7(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.reduce([[ARG0]] init: [[R0]]) applies stablehlo.add
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.subtract %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.multiply %arg5, %arg8 : tensor<i64>
    %7 = stablehlo.minimum %5, %6 : tensor<i64>
    %8 = stablehlo.maximum %5, %6 : tensor<i64>
    stablehlo.return %4, %7, %8 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       ├──────►│
//    │       │◄──────│
//    │◄──────┼───────┤
//    │       │       │
//    r0      r1      r2
//            U
//
// Non-conservative case. In theory r0 can be removed once r2 is removed, since
// they share a common dependency (a, a1). However, we cannot set synthetic
// return value, e.g. 0, instead of r2.

// CHECK-LABEL: func.func @reduce_unused_case8
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case8(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.subtract %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.multiply %arg5, %arg8 : tensor<i64>
    %7 = stablehlo.minimum %5, %6 : tensor<i64>
    %8 = stablehlo.maximum %5, %6 : tensor<i64>
    %9 = stablehlo.subtract %8, %4 : tensor<i64>
    stablehlo.return %4, %7, %9 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#1 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       │       │
//    ├───────┼──────►│
//    │       │       │
//    │       │       │
//    r0      r1      r2
//    U
//
// Non-conservative case: |{used results}| < |{used operand pairs}|.
//                                  |{r0}| < |{(a, a1), (c, c1)}|
// r1 can be dropped, but partial pruning is not implemented.

// CHECK-LABEL: func.func @reduce_unused_case9
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case9(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg5 : tensor<i64>
    %5 = stablehlo.subtract %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.multiply %arg5, %arg8 : tensor<i64>
    stablehlo.return %4, %5, %6 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴
//    │       │
//    │◄──────┼───────┐
//    │       │       │
//    │       │       │
//    r0      r1      r2
//    U               U
//
// Non-conservative case: |{used results}| > |{used operand pairs}|.
//                              |{r0, r2}| > |{(a, a1)}|
// r1 can be dropped, but partial pruning is not implemented.

// CHECK-LABEL: func.func @reduce_unused_case10
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case10(%arg0: tensor<8xi64>,
                                %arg1: tensor<8xi64>,
                                %arg2: tensor<8xi64>) -> (tensor<i64>, tensor<i64>) {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.subtract %arg4, %arg7 : tensor<i64>
    stablehlo.return %4, %5, %4 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0, %3#2 : tensor<i64>, tensor<i64>
}

// -----

/////////
// ReshapeOp

// CHECK-LABEL: func @reshape_identity
func.func @reshape_identity(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
  %0 = stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NOT: stablehlo.reshape
  // CHECK: return %arg0
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @reshape_reshape
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
func.func @reshape_reshape(%arg0: tensor<4x4xi32>) -> tensor<16xi32> {
  %0 = stablehlo.reshape %arg0 : (tensor<4x4xi32>) -> tensor<2x8xi32>
  %1 = stablehlo.reshape %0 : (tensor<2x8xi32>) -> tensor<16xi32>
  // CHECK: [[R0:%.+]] = stablehlo.reshape %[[ARG0]] : (tensor<4x4xi32>) -> tensor<16xi32>
  return %1 : tensor<16xi32>
}

// -----

/////////
// SubtractOp

// CHECK-LABEL: @subtract_same_lhs_rhs
func.func @subtract_same_lhs_rhs(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: stablehlo.constant dense<0> : tensor<2xi32>
  %0 = stablehlo.subtract %arg0, %arg0 : tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: @subtract_zero
func.func @subtract_zero(%arg0: tensor<2xi32>, %arg1: tensor<2xf32>) -> (tensor<2xi32>, tensor<2xf32>) {
  %0 = stablehlo.constant dense<0> : tensor<2xi32>
  %1 = stablehlo.subtract %arg0, %0 : tensor<2xi32>
  %2 = stablehlo.constant dense<0.0> : tensor<2xf32>
  %3 = stablehlo.subtract %arg1, %2 : tensor<2xf32>
  // CHECK-NOT: stablehlo.constant
  // CHECK: return %arg0, %arg1
  return %1, %3: tensor<2xi32>, tensor<2xf32>
}

// -----

/////////
// SelectOp

// CHECK-LABEL: func.func @select
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<2xi32>, [[ARGC:%.+]]: tensor<2xi1>)
func.func @select(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %argC: tensor<2xi1>)
  -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<4xi32>) {
  %c0 = stablehlo.constant dense<false> : tensor<i1>
  %c1 = stablehlo.constant dense<true> : tensor<i1>

  %c0_2 = stablehlo.constant dense<false> : tensor<2xi1>
  %c1_2 = stablehlo.constant dense<true> : tensor<2xi1>

  %cond = stablehlo.constant dense<[false, true, false, true]> : tensor<4xi1>
  %foo = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %bar = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi32>

  %0 = stablehlo.select %argC, %arg0, %arg0 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %1 = stablehlo.select %c0, %arg0, %arg1 : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %2 = stablehlo.select %c1, %arg0, %arg1 : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %3 = stablehlo.select %c0_2, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %4 = stablehlo.select %c1_2, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %5 = stablehlo.select %argC, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>

  %6 = stablehlo.select %cond, %foo, %bar : (tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.select [[ARGC]], [[ARG0]], [[ARG1]]
  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<[5, 2, 7, 4]> : tensor<4xi32>

  // CHECK-NEXT: return [[ARG0]], [[ARG1]], [[ARG0]], [[ARG1]], [[ARG0]], [[R0]], [[C0]]
  return %0, %1, %2, %3, %4, %5, %6 :
         tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<4xi32>
}

// CHECK-LABEL: func.func @select_into_minmax1
// CHECK-SAME:   [[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<2xi32>, [[ARG2:%.+]]: tensor<2xi32>, [[ARG3:%.+]]: tensor<2xi32>)
func.func @select_into_minmax1(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>,
                               %arg2: tensor<2xi32>, %arg3: tensor<2xi32>)
  -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) {

  %0 = stablehlo.compare EQ, %arg0, %arg1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %1 = stablehlo.compare NE, %arg0, %arg1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %2 = stablehlo.compare GE, %arg0, %arg1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %3 = stablehlo.compare GT, %arg0, %arg2, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %4 = stablehlo.compare LE, %arg1, %arg2, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %5 = stablehlo.compare LT, %arg1, %arg3, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>

  %s0 = stablehlo.select %0, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %s1 = stablehlo.select %1, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %s2 = stablehlo.select %2, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %s3 = stablehlo.select %3, %arg0, %arg2 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %s4 = stablehlo.select %4, %arg1, %arg2 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %s5 = stablehlo.select %5, %arg1, %arg3 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>

  // DISABLED-CHECK-DAG:  [[C0:%.+]] = stablehlo.compare EQ, [[ARG0]], [[ARG1]], SIGNED
  // DISABLED-CHECK-DAG:  [[C1:%.+]] = stablehlo.compare NE, [[ARG0]], [[ARG1]], SIGNED

  // DISABLED-CHECK-DAG:  [[S0:%.+]] = stablehlo.select [[C0]], [[ARG0]], [[ARG1]]
  // DISABLED-CHECK-DAG:  [[S1:%.+]] = stablehlo.select [[C1]], [[ARG0]], [[ARG1]]
  // DISABLED-CHECK-DAG:  [[S2:%.+]] = stablehlo.maximum [[ARG0]], [[ARG1]]
  // DISABLED-CHECK-DAG:  [[S3:%.+]] = stablehlo.maximum [[ARG0]], [[ARG2]]
  // DISABLED-CHECK-DAG:  [[S4:%.+]] = stablehlo.minimum [[ARG1]], [[ARG2]]
  // DISABLED-CHECK-DAG:  [[S5:%.+]] = stablehlo.minimum [[ARG1]], [[ARG3]]

  // DISABLED-CHECK-NEXT: return [[S0]], [[S1]], [[S2]], [[S3]], [[S4]], [[S5]]
  return %s0, %s1, %s2, %s3, %s4, %s5 :
         tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>
}

// CHECK-LABEL: func.func @select_into_minmax2
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<i32>, [[ARG1:%.+]]: tensor<i32>, [[ARG2:%.+]]: tensor<i32>, [[ARG3:%.+]]: tensor<i32>)
func.func @select_into_minmax2(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>)
  -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>,
      tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {

  %0 = stablehlo.compare GT, %arg1, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = stablehlo.compare GT, %arg1, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare GE, %arg1, %arg3, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare GE, %arg1, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %s0 = stablehlo.select %0, %arg0, %arg1 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s1 = stablehlo.select %1, %arg0, %arg1 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s2 = stablehlo.select %2, %arg3, %arg1 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s3 = stablehlo.select %3, %arg0, %arg2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>

  %4 = stablehlo.compare LT, %arg1, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %5 = stablehlo.compare LT, %arg0, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %6 = stablehlo.compare LE, %arg2, %arg3, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %7 = stablehlo.compare LE, %arg0, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %s4 = stablehlo.select %4, %arg2, %arg1 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s5 = stablehlo.select %5, %arg1, %arg2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s6 = stablehlo.select %6, %arg3, %arg2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s7 = stablehlo.select %7, %arg2, %arg3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>

  // DISABLED-CHECK-DAG:  [[C1:%.+]] = stablehlo.compare GT, [[ARG1]], [[ARG2]], SIGNED
  // DISABLED-CHECK-DAG:  [[C3:%.+]] = stablehlo.compare GE, [[ARG1]], [[ARG2]], SIGNED

  // DISABLED-CHECK-DAG:  [[S0:%.+]] = stablehlo.minimum [[ARG0]], [[ARG1]]
  // DISABLED-CHECK-DAG:  [[S1:%.+]] = stablehlo.select [[C1]], [[ARG0]], [[ARG1]]
  // DISABLED-CHECK-DAG:  [[S2:%.+]] = stablehlo.minimum [[ARG3]], [[ARG1]]
  // DISABLED-CHECK-DAG:  [[S3:%.+]] = stablehlo.select [[C3]], [[ARG0]], [[ARG2]]

  // DISABLED-CHECK-DAG:  [[C5:%.+]] = stablehlo.compare LT, [[ARG0]], [[ARG2]], SIGNED
  // DISABLED-CHECK-DAG:  [[C7:%.+]] = stablehlo.compare LE, [[ARG0]], [[ARG2]], SIGNED

  // DISABLED-CHECK-DAG:  [[S4:%.+]] = stablehlo.maximum [[ARG2]], [[ARG1]]
  // DISABLED-CHECK-DAG:  [[S5:%.+]] = stablehlo.select [[C5]], [[ARG1]], [[ARG2]]
  // DISABLED-CHECK-DAG:  [[S6:%.+]] = stablehlo.maximum [[ARG3]], [[ARG2]]
  // DISABLED-CHECK-DAG:  [[S7:%.+]] = stablehlo.select [[C7]], [[ARG2]], [[ARG3]]

  // DISABLED-CHECK-NEXT: return [[S0]], [[S1]], [[S2]], [[S3]], [[S4]], [[S5]], [[S6]], [[S7]]
  return %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>,
                                                  tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @select_op_not_as_pred(
func.func @select_op_not_as_pred(%arg0: tensor<4xi1>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.not %arg0 : tensor<4xi1>
  %1 = stablehlo.select %0, %arg1, %arg2 : tensor<4xi1>, tensor<4xf32>
  // CHECK-NOT: stablehlo.not
  // CHECK: %[[R:.*]] = stablehlo.select %arg0, %arg2, %arg1
  // CHECK: return %[[R]]
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @select_op_broadcasted_not_as_pred(
func.func @select_op_broadcasted_not_as_pred(%arg0: tensor<1xi1>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.not %arg0 : tensor<1xi1>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xi1>) -> tensor<4xi1>
  %2 = stablehlo.select %1, %arg1, %arg2 : tensor<4xi1>, tensor<4xf32>

  // CHECK-NOT: stablehlo.not
  // CHECK: %[[B:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xi1>) -> tensor<4xi1>
  // CHECK: %[[R:.*]] = stablehlo.select %[[B]], %arg2, %arg1
  // CHECK: return %[[R]]
  return %2 : tensor<4xf32>
}

// -----

/////////
// SliceOp

// CHECK-LABEL: slice_of_concat
// CHECK-SAME: [[ARG0:%.+]]: tensor<2x5xf32>, [[ARG1:%.+]]: tensor<1x5xf32>
func.func @slice_of_concat(%arg0: tensor<2x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<1x5xf32> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<2x5xf32>, tensor<1x5xf32>) -> tensor<3x5xf32>
  // CHECK-NOT: stablehlo.concatenate
  // CHECK: stablehlo.slice [[ARG0]]
  %1 = stablehlo.slice %0 [1:2, 0:5] : (tensor<3x5xf32>) -> tensor<1x5xf32>
  return %1 : tensor<1x5xf32>
}

// CHECK-LABEL: slice_2D_noop
// CHECK-SAME: [[ARG:%.+]]: tensor<2x2xi64>
func.func @slice_2D_noop(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
  %0 = stablehlo.slice %arg0 [0:2, 0:2] : (tensor<2x2xi64>) -> tensor<2x2xi64>

  // CHECK-NEXT: return [[ARG]]
  func.return %0 : tensor<2x2xi64>
}

// -----

/////////
// SortOp

// CHECK-LABEL: @sort_op_second_arg_unused
// CHECK-SAME: [[ARG0:%.+]]: tensor<3xi32>, [[ARG1:%.+]]: tensor<3xi32>
func.func @sort_op_second_arg_unused(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi32> {
  // CHECK: "stablehlo.sort"([[ARG0]])
  %0:2 = "stablehlo.sort"(%arg0, %arg1) <{dimension = 0 : i64, is_stable = false}> ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
    %1 = stablehlo.compare  GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  }) : (tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)
  return %0#0 : tensor<3xi32>
}

// CHECK-LABEL: @sort_op_set_default_dimension
func.func @sort_op_set_default_dimension(%arg0: tensor<3x5xi32>) -> tensor<3x5xi32> {
  // CHECK: stablehlo.sort{{.*}}dimension = 1 : i64
  %0 = "stablehlo.sort"(%arg0) <{dimension = -1 : i64, is_stable = false}> ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %1 = stablehlo.compare  GT, %arg1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  }) : (tensor<3x5xi32>) -> tensor<3x5xi32>
  return %0 : tensor<3x5xi32>
}

// -----

/////////
// TransposeOp

// CHECK-LABEL: @transpose_identity
func.func @transpose_identity(%arg0: tensor<2xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<f32>)
          -> (tensor<2xf32>, tensor<3x2xf32>, tensor<2x3xf32>, tensor<f32>) {
  %a = stablehlo.transpose %arg0, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
  %b = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<3x2xf32>) -> tensor<3x2xf32>
  %c = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
  %d = stablehlo.transpose %arg2, dims = [] : (tensor<f32>) -> tensor<f32>

  // CHECK-NEXT: [[X:%.+]] = stablehlo.transpose %arg1, dims = [1, 0]
  // CHECK-NEXT: return %arg0, %arg1, [[X]], %arg2
  return %a, %b, %c, %d : tensor<2xf32>, tensor<3x2xf32>, tensor<2x3xf32>, tensor<f32>
}

// CHECK-LABEL: @transpose_is_reshape
func.func @transpose_is_reshape(%arg0: tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32> {
  // CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32>
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] : (tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32>
  return %0 : tensor<1x4x1x5xf32>
}

// CHECK-LABEL: @transpose_is_not_reshape
func.func @transpose_is_not_reshape(%arg0: tensor<1x4x5x2xf32>) -> tensor<2x4x1x5xf32> {
  // CHECK-NOT: stablehlo.reshape
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] : (tensor<1x4x5x2xf32>) -> tensor<2x4x1x5xf32>
  return %0 : tensor<2x4x1x5xf32>
}

// CHECK-LABEL: @transpose_of_transpose
func.func @transpose_of_transpose(%arg0 : tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  %0 = stablehlo.transpose %arg0, dims = [3,2,1,0] : (tensor<1x2x3x4xf32>) -> tensor<4x3x2x1xf32>
  %1 = stablehlo.transpose %0, dims = [3,2,1,0] : (tensor<4x3x2x1xf32>) -> tensor<1x2x3x4xf32>
  // CHECK-NOT: stablehlo.transpose
  // CHECK: return %arg0
  return %1 : tensor<1x2x3x4xf32>
}

// -----

////////
// TupleOp


// CHECK-LABEL: unpack_repack_same_tuple
// CHECK-SAME: ([[ARG0:%.*]]: tuple<tensor<i32>, !stablehlo.token, tensor<f32>>)
func.func @unpack_repack_same_tuple(%arg0: tuple<tensor<i32>, !stablehlo.token, tensor<f32>>) -> tuple<tensor<i32>, !stablehlo.token, tensor<f32>> {
  %0 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<i32>, !stablehlo.token, tensor<f32>>) -> tensor<i32>
  %1 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<i32>, !stablehlo.token, tensor<f32>>) -> !stablehlo.token
  %2 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<i32>, !stablehlo.token, tensor<f32>>) -> tensor<f32>
  %3 = stablehlo.tuple %0, %1, %2 : tuple<tensor<i32>, !stablehlo.token, tensor<f32>>
  // CHECK: return [[ARG0]]
  return %3 : tuple<tensor<i32>, !stablehlo.token, tensor<f32>>
}

// CHECK-LABEL: unpack_repack_same_tuple_single_element
// CHECK-SAME: ([[ARG0:%.*]]: tuple<tensor<i32>>)
func.func @unpack_repack_same_tuple_single_element(%arg0: tuple<tensor<i32>>) -> tuple<tensor<i32>> {
  %0 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<i32>>) -> tensor<i32>
  %1 = stablehlo.tuple %0 : tuple<tensor<i32>>
  // CHECK: return [[ARG0]]
  return %1 : tuple<tensor<i32>>
}

// -----

////////
// WhileOp

// CHECK-LABEL: while_op_with_outfeed_no_dce
func.func @while_op_with_outfeed_no_dce(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK: stablehlo.while
  %0 = stablehlo.while(%iterArg = %arg0) : tensor<i64>
    cond {
    %1 = stablehlo.compare  LT, %iterArg, %iterArg : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.create_token : !stablehlo.token
    %2 = "stablehlo.outfeed"(%iterArg, %1) <{outfeed_config = ""}> : (tensor<i64>, !stablehlo.token) -> !stablehlo.token
    stablehlo.return %iterArg : tensor<i64>
  }
  return %arg0 : tensor<i64>
}

// CHECK-LABEL: while_op_dce_no_side_effect
func.func @while_op_dce_no_side_effect(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK-NOT: stablehlo.while
  %0 = stablehlo.while(%iterArg = %arg0) : tensor<i64>
    cond {
    %1 = stablehlo.compare  LT, %iterArg, %iterArg : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.create_token : !stablehlo.token
    stablehlo.return %iterArg : tensor<i64>
  }
  return %arg0 : tensor<i64>
}

// Constant capture
// CHECK-LABEL: while_op_constant_capture
func.func @while_op_constant_capture(%arg0: tensor<10xf32>) -> (tensor<10xf32>) {
  %c = stablehlo.constant dense<1> : tensor<i32>
  %c_0 = stablehlo.constant dense<10> : tensor<i32>
  %c_1 = stablehlo.constant dense<0> : tensor<i32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<10xf32>
  // CHECK: stablehlo.while(%iterArg = %c_1, %iterArg_2 = %0) : tensor<i32>, tensor<10xf32> attributes {mhlo.frontend_attributes = {test_attr = "true"}}
  %1:3 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c_1, %iterArg_3 = %0) : tensor<10xf32>, tensor<i32>, tensor<10xf32> attributes {mhlo.frontend_attributes = {test_attr = "true"}}
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  } do {
    %2 = stablehlo.dynamic_slice %iterArg, %iterArg_2, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.sine %3 : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.dynamic_update_slice %iterArg_3, %5, %iterArg_2 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
    %7 = stablehlo.add %iterArg_2, %c : tensor<i32>
    stablehlo.return %iterArg, %7, %6 : tensor<10xf32>, tensor<i32>, tensor<10xf32>
  }
  return %1#2 : tensor<10xf32>
}

// -----

/////////
// Generic Zero Extent Ops

// CHECK-LABEL: func.func @reduce_zero_ext
func.func @reduce_zero_ext(%arg0: tensor<0xi1>) -> tensor<i32> {
  %0 = stablehlo.constant dense<false> : tensor<i1>
  %1 = stablehlo.constant dense<false> : tensor<0xi1>
  %2 = stablehlo.compare NE, %arg0, %1, UNSIGNED : (tensor<0xi1>, tensor<0xi1>) -> tensor<0xi1>
  %3 = stablehlo.convert %2 : (tensor<0xi1>) -> tensor<0xi32>
  %4 = stablehlo.constant dense<0> : tensor<i32>
  %5 = stablehlo.reduce(%3 init: %4) across dimensions = [0] : (tensor<0xi32>, tensor<i32>) -> tensor<i32>
    reducer(%arg1: tensor<i32>, %arg2: tensor<i32>)  {
    %6 = stablehlo.add %arg1, %arg2 : tensor<i32>
    stablehlo.return %6 : tensor<i32>
  }

  // CHECK: [[CST:%.+]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: return [[CST]] : tensor<i32>
  return %5 : tensor<i32>
}

// -----

/////////
// XorOp

// CHECK-LABEL: @xor_cst_on_rhs
func.func @xor_cst_on_rhs(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %cst = stablehlo.constant dense<false> : tensor<2xi1>
  %0 = stablehlo.xor %cst, %arg0 : tensor<2xi1>
  // CHECK: stablehlo.xor %arg0, %c : tensor<2xi1>
  return %0 : tensor<2xi1>
}

// -----

/////////
// Zero Extents

// CHECK-LABEL: func.func @add_zero_ext
func.func @add_zero_ext(%arg0 : tensor<5x0xi32>, %arg1 : tensor<5x0xi32>) -> tensor<5x0xi32> {
  // CHECK:   %[[EMPTY:.+]] = stablehlo.constant dense<>
  // CHECK:   return %[[EMPTY]]
  %0 = stablehlo.add %arg0, %arg1 : tensor<5x0xi32>
  func.return %0 : tensor<5x0xi32>
}

// -----

// CHECK-LABEL: func.func @add_zero_ext_dynamic
func.func @add_zero_ext_dynamic(%arg0 : tensor<?x0xi32>, %arg1 : tensor<?x0xi32>) -> tensor<?x0xi32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<?x0xi32>
  // CHECK-NOT:   stablehlo.constant dense<>
  func.return %0 : tensor<?x0xi32>
}

// -----

// CHECK-LABEL: func.func @scatter_zero_ext
func.func @scatter_zero_ext(%arg0 : tensor<f32>, %arg1 : tensor<1x0xi32>, %arg2 : tensor<1xf32>) -> tensor<f32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<f32>, tensor<1x0xi32>, tensor<1xf32>) -> tensor<f32>
  // CHECK:   %[[EMPTY:.+]] = stablehlo.constant dense<> : tensor<1x0xi32>
  // CHECK:   %[[SCATTER:.+]] = "stablehlo.scatter"(%arg0, %[[EMPTY]], %arg2)
  // CHECK:   return %[[SCATTER]]
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: slice_zero_extent
func.func @slice_zero_extent(%arg0: tensor<1x5xf32>) -> tensor<0x5xf32> {
  %0 = stablehlo.slice %arg0 [1:1, 0:5] : (tensor<1x5xf32>) -> tensor<0x5xf32>
  // CHECK-NOT: stablehlo.slice
  // CHECK: [[CST:%.+]] = stablehlo.constant dense<> : tensor<0x5xf32>
  // CHECK: return [[CST]]
  return %0 : tensor<0x5xf32>
}

// -----

// CHECK-LABEL: @sort_zero_extent
func.func public @sort_zero_extent(%arg0: tensor<0xi16>) -> (tensor<0xi32> {jax.result_info = ""}) {
  %0 = stablehlo.iota dim = 0 : tensor<0xi32>
  %1:2 = "stablehlo.sort"(%arg0, %0) ({
  ^bb0(%arg1: tensor<i16>, %arg2: tensor<i16>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %2 = stablehlo.compare  LT, %arg1, %arg2,  SIGNED : (tensor<i16>, tensor<i16>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<0xi16>, tensor<0xi32>) -> (tensor<0xi16>, tensor<0xi32>)

  // CHECK: %[[EMPTY:.+]] = stablehlo.constant dense<> : tensor<0xi32>
  // CHECK: return %[[EMPTY]]
  return %1#1 : tensor<0xi32>
}


// -----

// CHECK-LABEL: @while_zero_extent
// CHECK: %[[R0:.+]] = stablehlo.constant dense<> : tensor<75x0xf32>
// CHECK: %[[R2:.+]] = stablehlo.while
// CHECK: return %[[R2]], %[[R0]]


func.func public @while_zero_extent(%arg0: tensor<i32>, %arg1: tensor<3xf32>, %arg2: tensor<75x0xf32>) -> (tensor<i32>, tensor<75x0xf32>) {
  %0 = stablehlo.constant dense<1> : tensor<i32>
  %1 = stablehlo.constant dense<75> : tensor<i32>
  %2 = stablehlo.constant dense<0> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %2, %iterArg_2 = %arg2) : tensor<i32>, tensor<75x0xf32>
   cond {
    %4 = stablehlo.compare  LT, %iterArg, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %44 = stablehlo.add %iterArg, %0 : tensor<i32>
    stablehlo.return %44, %iterArg_2 : tensor<i32>, tensor<75x0xf32>
  }
  return %3#0, %3#1 : tensor<i32>, tensor<75x0xf32>
}

// -----

// CHECK-LABEL: @side_effecting_custom_call
func.func @side_effecting_custom_call(%arg0: tensor<0xf32>) -> (tensor<0xf32>, tensor<0xf32>) {
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<> : tensor<0xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @foo(%arg0) {api_version = 0 : i32, has_side_effect = true} : (tensor<0xf32>) -> tensor<0xf32>
  %0 = stablehlo.custom_call @foo(%arg0) {api_version = 0 : i32, has_side_effect = true} : (tensor<0xf32>) -> tensor<0xf32>
  // CHECK-NOT:  stablehlo.custom_call{{.*}}has_side_effect = false
  %1 = stablehlo.custom_call @foo(%arg0) {api_version = 0 : i32, has_side_effect = false} : (tensor<0xf32>) -> tensor<0xf32>
  // CHECK: return %[[CC]], %[[CST]]
  return %0, %1 : tensor<0xf32>, tensor<0xf32>
}

// -----

/////////
// Generic Shape Ops

// CHECK-LABEL: @push_shape_ops_to_end
func.func @push_shape_ops_to_end(%arg0 : tensor<12xf32>) -> tensor<3x4x2x1xf32> {
  // DISABLED-CHECK: %[[COS:.+]] = stablehlo.cosine %arg0 : tensor<12xf32>
  // DISABLED-CHECK: %[[ABS:.+]] = stablehlo.abs %[[COS]] : tensor<12xf32>
  // DISABLED-CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %[[ABS]] : (tensor<12xf32>) -> tensor<3x4xf32>
  // DISABLED-CHECK: %[[BROADCAST:.+]] = stablehlo.broadcast %[[RESHAPE]], sizes = [1, 2] : (tensor<3x4xf32>) -> tensor<1x2x3x4xf32>
  // DISABLED-CHECK: %[[TRANSPOSE:.+]] = stablehlo.transpose %[[BROADCAST]], dims = [2, 3, 1, 0] : (tensor<1x2x3x4xf32>) -> tensor<3x4x2x1xf32>
  // DISABLED-CHECK: return %[[TRANSPOSE]]
  %0 = stablehlo.reshape %arg0 : (tensor<12xf32>) -> tensor<3x4xf32>
  %1 = stablehlo.broadcast %0, sizes = [1, 2] : (tensor<3x4xf32>) -> tensor<1x2x3x4xf32>
  %2 = stablehlo.cosine %1 : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %3 = stablehlo.transpose %2, dims = [2, 3, 1, 0]  : (tensor<1x2x3x4xf32>) -> tensor<3x4x2x1xf32>
  %4 = stablehlo.abs %3 : (tensor<3x4x2x1xf32>) -> tensor<3x4x2x1xf32>
  return %4 : tensor<3x4x2x1xf32>
}


// -----

// CHECK-LABEL: @reorder_with_type_change
func.func @reorder_with_type_change(%arg0 : tensor<3x4xi32>) -> tensor<12xi64> {
  // DISABLED-CHECK: %[[CONVERT:.+]] = stablehlo.convert %arg0 : (tensor<3x4xi32>) -> tensor<3x4xi64>
  // DISABLED-CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %[[CONVERT]] : (tensor<3x4xi64>) -> tensor<12xi64>
  // DISABLED-CHECK: return %[[RESHAPE]]
  %0 = stablehlo.reshape %arg0 : (tensor<3x4xi32>) -> tensor<12xi32>
  %1 = stablehlo.convert %0 : (tensor<12xi32>) -> tensor<12xi64>
  return %1 : tensor<12xi64>
}

// -----

// CHECK-LABEL: @reorder_invalid_with_dynamic_shape
func.func @reorder_invalid_with_dynamic_shape(%arg0: tensor<1x3x4xf32>) -> (tensor<?x4xf32>) {
  // CHECK:      %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
  // CHECK-NEXT: %[[CONVERT:.+]] = stablehlo.convert %[[RESHAPE]] : (tensor<3x4xf32>) -> tensor<?x4xf32>
  // CHECK: return %[[CONVERT]]
  %0 = stablehlo.reshape %arg0 : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
  %1 = stablehlo.convert %0 : (tensor<3x4xf32>) -> tensor<?x4xf32>
  return %1 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @do_not_reorder_with_other_uses
func.func @do_not_reorder_with_other_uses(%arg0: tensor<2x2xf64>, %arg1: tensor<4xf32>, %arg2: tensor<f64>) -> (tensor<f64>, tensor<4xf32>) {
  // CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<2x2xf64>) -> tensor<4xf64>
  // CHECK: %[[CONVERT:.+]] = stablehlo.convert %[[RESHAPE]] : (tensor<4xf64>) -> tensor<4xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<2x2xf64>) -> tensor<4xf64>
  %1 = stablehlo.convert %0 : (tensor<4xf64>) -> tensor<4xf32>
  %2 = stablehlo.subtract %arg1, %1 : tensor<4xf32>
  %3 = stablehlo.reduce(%0 init: %arg2) across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
    reducer(%arg3: tensor<f64>, %arg4: tensor<f64>)  {
    %4 = stablehlo.add %arg3, %arg4 : tensor<f64>
    stablehlo.return %4 : tensor<f64>
  }
  return %3, %2 : tensor<f64>, tensor<4xf32>
}


// -----

// Make sure we do not crash on unregistered dialects.

// CHECK-LABEL: func.func @generic_op
func.func @generic_op(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NEXT:    "test_dialect.op"
  // CHECK-NEXT:    return
  %0 = "test_dialect.op"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>)
  return %0 : tensor<2xf32>
}
