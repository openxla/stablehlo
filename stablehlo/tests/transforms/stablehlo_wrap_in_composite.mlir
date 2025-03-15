// RUN: stablehlo-opt --stablehlo-wrap-in-composite='op-names=stablehlo.add,stablehlo.convolution,stablehlo.reduce version=1' --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @wrap_in_composite
// CHECK-SAME: %[[ARG_0:.*]]: tensor<64x8x8x8xi8>,
// CHECK-SAME: %[[ARG_1:.*]]: tensor<4x4x8x32xi8>,
// CHECK-SAME: %[[ARG_2:.*]]: tensor<64x3x3x32xi32>) -> tensor<64x3x3x32xi32> {
// CHECK: %[[CONV:.*]] = stablehlo.composite "stablehlo.convolution" %[[ARG_0]], %[[ARG_1]] {
// CHECK-SAME: composite_attributes = {batch_group_count = 1 : i64,
// CHECK-SAME: dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
// CHECK-SAME: feature_group_count = 1 : i64,
// CHECK-SAME{LITERAL}: padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
// CHECK-SAME{LITERAL}: rhs_dilation = array<i64: 2, 2>,
// CHECK-SAME{LITERAL}: window_strides = array<i64: 1, 1>},
// CHECK-SAME: decomposition = @stablehlo.convolution.impl,
// CHECK-SAME: version = 1 : i32} : (tensor<64x8x8x8xi8>, tensor<4x4x8x32xi8>) -> tensor<64x3x3x32xi32>
// CHECK: %[[ADD:.*]] = stablehlo.composite "stablehlo.add" %[[CONV]], %[[ARG_2]] {
// CHECK-SAME: decomposition = @stablehlo.add.impl,
// CHECK-SAME: version = 1 : i32} : (tensor<64x3x3x32xi32>, tensor<64x3x3x32xi32>) -> tensor<64x3x3x32xi32>
// CHECK-NEXT: return %[[ADD]]

// CHECK-LABEL: func.func private @stablehlo.add.impl
// CHECK-SAME: %[[ARG_0:.*]]: tensor<64x3x3x32xi32>,
// CHECK-SAME: %[[ARG_1:.*]]: tensor<64x3x3x32xi32>) -> tensor<64x3x3x32xi32> {
// CHECK: %[[VAL:.*]] = stablehlo.add %[[ARG_0]], %[[ARG_1]] : tensor<64x3x3x32xi32>
// CHECK-NEXT: return %[[VAL]]

// CHECK-LABEL: func.func private @stablehlo.convolution.impl
// CHECK-SAME: %[[ARG_0:.*]]: tensor<64x8x8x8xi8>,
// CHECK-SAME: %[[ARG_1:.*]]: tensor<4x4x8x32xi8>) -> tensor<64x3x3x32xi32> {
// CHECK: %[[VAL:.*]] = stablehlo.convolution(%[[ARG_0]], %[[ARG_1]])
// CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
// CHECK-SAME{LITERAL}: stride = [1, 1],
// CHECK-SAME{LITERAL}: pad = [[0, 1], [0, 1]],
// CHECK-SAME{LITERAL}: rhs_dilate = [2, 2]}
// CHECK-SAME: batch_group_count = 1 : i64
// CHECK-SAME: feature_group_count = 1 : i64
// CHECK-SAME: : (tensor<64x8x8x8xi8>, tensor<4x4x8x32xi8>) -> tensor<64x3x3x32xi32>
// CHECK-NEXT: return %[[VAL]]

func.func @wrap_in_composite(
    %arg0: tensor<64x8x8x8xi8>,
    %arg1: tensor<4x4x8x32xi8>,
    %arg2: tensor<64x3x3x32xi32>) -> tensor<64x3x3x32xi32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [2, 2]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64} :
       (tensor<64x8x8x8xi8>, tensor<4x4x8x32xi8>) -> tensor<64x3x3x32xi32>
  %1 = stablehlo.add %0, %arg2 : tensor<64x3x3x32xi32>
  func.return %1 : tensor<64x3x3x32xi32>
}

// -----

// CHECK-LABEL: func.func @wrap_in_composite_op_with_region
// CHECK-SAME: %[[ARG_0:.*]]: tensor<4x3xf32>) -> tensor<4xf32>
// CHECK: %[[CONST:.*]] = stablehlo.constant
// CHECK-NEXT: %[[COMPOSITE_REDUCE:.*]] = stablehlo.composite "stablehlo.reduce" %[[ARG_0]], %[[CONST]] {
// CHECK-SAME: composite_attributes = {
// CHECK-SAME: dimensions = array<i64: 1>},
// CHECK-SAME: decomposition = @stablehlo.reduce.impl,
// CHECK-SAME: version = 1 : i32} : (tensor<4x3xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[COMPOSITE_REDUCE]]

// CHECK-LABEL: func.func private @stablehlo.reduce.impl
// CHECK-SAME: %[[ARG_0:.*]]: tensor<4x3xf32>,
// CHECK-SAME: %[[ARG_1:.*]]: tensor<f32>) -> tensor<4xf32> {
// CHECK: %[[REDUCE:.*]] = stablehlo.reduce(%[[ARG_0]] init: %[[ARG_1]])
// CHECK-SAME{LITERAL}: applies stablehlo.add across dimensions = [1]
// CHECK-SAME: (tensor<4x3xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[REDUCE]]
func.func @wrap_in_composite_op_with_region(%x : tensor<4x3xf32>) -> tensor<4xf32> {
  %cst = stablehlo.constant dense<2.7> : tensor<f32>
  %res = stablehlo.reduce(%x init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x3xf32>, tensor<f32>) -> tensor<4xf32>
  func.return %res : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @cannot_be_wrapped_ops_does_not_match
// CHECK-SAME: %[[ARG_0:.*]]: tensor<2xf32>,
// CHECK-SAME: %[[ARG_1:.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK: %[[VAL:.*]] = stablehlo.multiply %[[ARG_0]], %[[ARG_1]] : tensor<2xf32>
// CHECK-NEXT: return %[[VAL]] : tensor<2xf32>
func.func @cannot_be_wrapped_ops_does_not_match(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<2xf32>
  func.return %0 : tensor<2xf32>
}
