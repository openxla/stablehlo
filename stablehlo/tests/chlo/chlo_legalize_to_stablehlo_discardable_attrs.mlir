// RUN: stablehlo-opt --chlo-legalize-to-stablehlo --split-input-file %s | FileCheck %s

// Discardable attributes (e.g. frontend metadata) on a CHLO op are carried over
// to the StableHLO op that directly replaces it.

// CHECK-LABEL: func.func @broadcast_add_no_broadcast_discardable_attrs
func.func @broadcast_add_no_broadcast_discardable_attrs(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  //      CHECK: stablehlo.add %arg0, %arg1
  // CHECK-SAME:   {mhlo.frontend_attributes = {MUST_FUSE = "0"}}
  %0 = chlo.broadcast_add %arg0, %arg1 {mhlo.frontend_attributes = {MUST_FUSE = "0"}} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_add_static_broadcast_discardable_attrs
func.func @broadcast_add_static_broadcast_discardable_attrs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  //      CHECK: stablehlo.broadcast_in_dim
  //  CHECK-NOT:   mhlo.frontend_attributes
  //      CHECK: stablehlo.add
  // CHECK-SAME:   {mhlo.frontend_attributes = {MUST_FUSE = "0"}}
  %0 = chlo.broadcast_add %arg0, %arg1 {mhlo.frontend_attributes = {MUST_FUSE = "0"}} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_add_dynamic_broadcast_discardable_attrs
func.func @broadcast_add_dynamic_broadcast_discardable_attrs(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //      CHECK: shape.assuming
  //  CHECK-NOT:   mhlo.frontend_attributes
  //      CHECK:   stablehlo.add
  // CHECK-SAME:     {mhlo.frontend_attributes = {MUST_FUSE = "0"}}
  %0 = chlo.broadcast_add %arg0, %arg1 {mhlo.frontend_attributes = {MUST_FUSE = "0"}} : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_compare_discardable_attrs
func.func @broadcast_compare_discardable_attrs(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  //      CHECK: stablehlo.compare EQ, %arg0, %arg1
  // CHECK-SAME:   {mhlo.frontend_attributes = {MUST_FUSE = "0"}}
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #chlo<comparison_direction EQ>, mhlo.frontend_attributes = {MUST_FUSE = "0"}} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: func.func @broadcast_select_discardable_attrs
func.func @broadcast_select_discardable_attrs(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  //      CHECK: stablehlo.select
  // CHECK-SAME:   {mhlo.frontend_attributes = {MUST_FUSE = "0"}}
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) {mhlo.frontend_attributes = {MUST_FUSE = "0"}} : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func.func @constant_discardable_attrs
func.func @constant_discardable_attrs() -> tensor<2xf32> {
  //      CHECK: stablehlo.constant
  // CHECK-SAME:   {mhlo.frontend_attributes = {MUST_FUSE = "0"}}
  %0 = "chlo.constant"() {value = dense<1.0> : tensor<2xf32>, mhlo.frontend_attributes = {MUST_FUSE = "0"}} : () -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @ragged_dot_mode_3_discardable_attrs
func.func @ragged_dot_mode_3_discardable_attrs(%lhs : tensor<2x3x5xf32>, %rhs : tensor<2x5x7xf32>, %group_sizes : tensor<2xi64>) -> tensor<2x3x7xf32> {
  //      CHECK: stablehlo.dot_general
  // CHECK-SAME:   mhlo.frontend_attributes = {MUST_FUSE = "0"}
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1],
      lhs_ragged_dimensions = [0],
      rhs_group_dimensions = []
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>],
    mhlo.frontend_attributes = {MUST_FUSE = "0"}
  } : (tensor<2x3x5xf32>, tensor<2x5x7xf32>, tensor<2xi64>) -> tensor<2x3x7xf32>
  func.return %0 : tensor<2x3x7xf32>
}
