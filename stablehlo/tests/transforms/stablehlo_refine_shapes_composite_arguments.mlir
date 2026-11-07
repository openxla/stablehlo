// RUN: stablehlo-opt %s --stablehlo-refine-arguments='types=tensor<512xi32>' --stablehlo-refine-shapes | FileCheck %s
// RUN: stablehlo-opt %s --stablehlo-refine-arguments='types=tensor<512xi32>' --stablehlo-refine-shapes --stablehlo-refine-shapes | FileCheck %s

// Regression test for https://github.com/openxla/stablehlo/issues/2779.
// The outer call loses its dimension argument, but the composite and its
// decomposition must retain both operands and agree on their refined types.
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[ARG:.*]]: tensor<512xi32>) -> tensor<512xi32>
// CHECK: call @main_wrapped(%[[ARG]]) : (tensor<512xi32>) -> tensor<512xi32>
func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %1 = call @main_wrapped(%0, %arg0) : (tensor<i32>, tensor<?xi32>) -> tensor<?xi32>
  return %1 : tensor<?xi32>
}

// CHECK-LABEL: func.func @main_wrapped
// CHECK-SAME: (%[[ARG:.*]]: tensor<512xi32>) -> tensor<512xi32>
// CHECK: %[[DIM:.*]] = stablehlo.constant dense<512> : tensor<i32>
// CHECK: stablehlo.composite "my.mul2" %[[DIM]], %[[ARG]] {decomposition = @my.mul2.impl} : (tensor<i32>, tensor<512xi32>) -> tensor<512xi32>
func.func @main_wrapped(%arg0: tensor<i32> {jax.global_constant = "a"}, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  %0 = stablehlo.composite "my.mul2" %arg0, %arg1 {decomposition = @my.mul2.impl} : (tensor<i32>, tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func private @my.mul2.impl
// CHECK-SAME: (%{{.*}}: tensor<i32> {jax.global_constant = "a"}, %{{.*}}: tensor<512xi32>) -> tensor<512xi32>
// CHECK: stablehlo.multiply {{.*}} : tensor<512xi32>
func.func private @my.mul2.impl(%arg0: tensor<i32> {jax.global_constant = "a"}, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  %c_2 = stablehlo.constant dense<2> : tensor<i32>
  %shape = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
  %0 = stablehlo.dynamic_broadcast_in_dim %c_2, %shape, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
  %1 = stablehlo.multiply %arg1, %0 : tensor<?xi32>
  return %1 : tensor<?xi32>
}
