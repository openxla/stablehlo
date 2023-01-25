// RUN: stablehlo-opt --hlo-test-infer --allow-unregistered-dialect --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @broadcast_add_reify
// Note that all broadcast_ops are expanded from the same template, so
// only test reification on an examplar op.
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<?xf32>
func.func @broadcast_add_reify(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<1xindex> {
  // CHECK-DAG: %[[ARG0_S:.+]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[ARG1_S:.+]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[BCAST_S:.+]] = shape.broadcast %[[ARG0_S]], %[[ARG1_S]] : tensor<1xindex>, tensor<1xindex> -> tensor<1xindex>
  // CHECK: return %[[BCAST_S]] : tensor<1xindex>
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "hlo_test_infer.reify_return_type_shapes"(%0) : (tensor<?xf32>) -> tensor<1xindex>
  func.return %1 : tensor<1xindex>
}

// -----
// CHECK-LABEL: @broadcast_add_different_operand_size
func.func @broadcast_add_different_operand_size(%arg1: tensor<1xi32>, %arg2: tensor<1x2xi32>) -> tensor<1x2xi32> {
  %0 = "chlo.broadcast_add"(%arg1, %arg2) {broadcast_dimensions = dense<1> : tensor<i64>} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  // CHECK: types0 = tensor<1x2xi32>
  %1 = "hlo_test_infer.get_return_types"(%0) : (tensor<1x2xi32>) -> tensor<1x2xi32>
  return %1: tensor<1x2xi32>
}

// -----
// CHECK-LABEL: @broadcast_complex_ranked_components
func.func @broadcast_complex_ranked_components(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>> {
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  // CHECK: types0 = tensor<?x?xcomplex<f32>>
  %1 = "hlo_test_infer.get_return_types"(%0) : (tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>
  func.return %1 : tensor<?x?xcomplex<f32>>
}

// -----
// CHECK-LABEL: @broadcast_complex_reify
func.func @broadcast_complex_reify(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<2xindex> {
  // CHECK:      %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  // CHECK-NEXT: %1 = shape.shape_of %arg1 : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK-NEXT: %2 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<2xindex> -> tensor<2xindex>
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  %1 = "hlo_test_infer.reify_return_type_shapes"(%0) : (tensor<?x?xcomplex<f32>>) -> tensor<2xindex>
  func.return %1 : tensor<2xindex>
}

// -----
func.func @broadcast_complex_mismatch(%arg0: tensor<2xf64>, %arg1: tensor<2xf32>) -> tensor<2xcomplex<f32>> {
  // expected-error @+1 {{mismatched operand types}}
  %0 = "chlo.broadcast_complex"(%arg0, %arg1) : (tensor<2xf64>, tensor<2xf32>) -> tensor<2xcomplex<f32>>
  %1 = "hlo_test_infer.get_return_types"(%0) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  return %0: tensor<2xcomplex<f32>>
}

// -----
// CHECK-LABEL: @broadcast_compare_ranked_components
func.func @broadcast_compare_ranked_components(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xi1> {
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #chlo<comparison_direction EQ>} : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  // CHECK: types0 = tensor<?x?xi1>
  %1 = "hlo_test_infer.get_return_types"(%0) : (tensor<?x?xi1>) -> tensor<?x?xi1>
  func.return %0 : tensor<?x?xi1>
}

// -----
// CHECK-LABEL: @broadcast_compare_unranked_reify
func.func @broadcast_compare_unranked_reify(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<?xindex> {
  // expected-warning @+1 {{unsupported non prefix-padded dynamic rank broadcast_dimensions = dense<1> : tensor<1xi64>}}
  %0 = chlo.broadcast_compare %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = #chlo<comparison_direction EQ>} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xi1>
  %1 = "hlo_test_infer.reify_return_type_shapes"(%0) : (tensor<*xi1>) -> tensor<?xindex>
  func.return %1 : tensor<?xindex>
}

// -----
// CHECK-LABEL: @broadcast_compare_reify
func.func @broadcast_compare_reify(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<2xindex> {
  // CHECK:      %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  // CHECK-NEXT: %1 = shape.shape_of %arg1 : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK-NEXT: %2 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<2xindex> -> tensor<2xindex>
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #chlo<comparison_direction EQ>} : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  %1 = "hlo_test_infer.reify_return_type_shapes"(%0) : (tensor<?x?xi1>) -> tensor<2xindex>
  func.return %1 : tensor<2xindex>
}

// -----
// CHECK-LABEL: @broadcast_add_ranked_components_r1
func.func @broadcast_add_ranked_components_r1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<1xindex> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "hlo_test_infer.reify_return_type_shapes"(%0) : (tensor<?xf32>) -> tensor<1xindex>
  func.return %1 : tensor<1xindex>
}

// -----
// CHECK-LABEL: @broadcast_add_ranked_components_r1x2
func.func @broadcast_add_ranked_components_r1x2(%arg0: tensor<?xf32>, %arg1: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
  // CHECK: types0 = tensor<?x3xf32>
  %1 = "hlo_test_infer.get_return_types"(%0) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  func.return %1 : tensor<?x3xf32>
}

// -----
// CHECK-LABEL: @broadcast_add_ranked_components_with_zero_r1x2
func.func @broadcast_add_ranked_components_with_zero_r1x2(%arg0: tensor<0xf32>, %arg1: tensor<?x1xf32>) -> tensor<?x0xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<0xf32>, tensor<?x1xf32>) -> tensor<?x0xf32>
  // CHECK: types0 = tensor<?x0xf32>
  %1 = "hlo_test_infer.get_return_types"(%0) : (tensor<?x0xf32>) -> tensor<?x0xf32>
  func.return %1 : tensor<?x0xf32>
}

// -----
func.func @broadcast_select_branch_mismatch(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xf32>) -> tensor<2xi32> {
  // expected-error @+1 {{mismatched operand types}}
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xf32>) -> tensor<2xi32>
  %1 = "hlo_test_infer.get_return_types"(%0) : (tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// -----
// CHECK-LABEL: @broadcast_select_reify
func.func @broadcast_select_reify(%arg0: tensor<2xi1>, %arg1: tensor<?xi32>, %arg2: tensor<?xi32>) -> tensor<1xindex> {
  // CHECK:      %0 = shape.const_shape [2] : tensor<1xindex>
  // CHECK-NEXT: %1 = shape.shape_of %arg1 : tensor<?xi32> -> tensor<1xindex>
  // CHECK-NEXT: %2 = shape.shape_of %arg2 : tensor<?xi32> -> tensor<1xindex>
  // CHECK-NEXT: %3 = shape.broadcast %1, %2, %0 : tensor<1xindex>, tensor<1xindex>, tensor<1xindex> -> tensor<1xindex>
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %1 = "hlo_test_infer.reify_return_type_shapes"(%0) : (tensor<?xi32>) -> tensor<1xindex>
  return %1: tensor<1xindex>
}

// -----
// CHECK-LABEL: @constant_ranked
func.func @constant_ranked() -> (tensor<i32>) {
  %0 = "chlo.constant"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  // CHECK: types0 = tensor<i32>
  %1 = "hlo_test_infer.get_return_types"(%0) : (tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// -----
// CHECK-LABEL: @constant_like_ranked
func.func @constant_like_ranked(%arg0: tensor<1x?xi64>) -> (tensor<1x?xf32>) {
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<1x?xi64>) -> tensor<1x?xf32>
  // CHECK: types0 = tensor<1x?xf32>
  %1 = "hlo_test_infer.get_return_types"(%0) : (tensor<1x?xf32>) -> tensor<1x?xf32>
  func.return %1 : tensor<1x?xf32>
}

// -----
// CHECK-LABEL: @constant_like_unranked
func.func @constant_like_unranked(%arg0: tensor<*xi64>) -> (tensor<*xf32>) {
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<*xi64>) -> tensor<*xf32>
  // CHECK: types0 = tensor<*xf32>
  %1 = "hlo_test_infer.get_return_types"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----
// CHECK-LABEL: @constant_like_reify
func.func @constant_like_reify(%arg0: tensor<?xi64>) -> (tensor<1xindex>) {
  // CHECK: shape.shape_of %arg0 : tensor<?xi64> -> tensor<1xindex>
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<?xi64>) -> tensor<?xf32>
  %1 = "hlo_test_infer.reify_return_type_shapes"(%0) : (tensor<?xf32>) -> tensor<1xindex>
  func.return %1 : tensor<1xindex>
}

// -----
// CHECK-LABEL: @is_inf_ops_return_types
func.func @is_inf_ops_return_types(%arg : tensor<f32>) -> tensor<i1> {
  %0 = chlo.is_inf %arg : tensor<f32> -> tensor<i1>
  %1 = chlo.is_neg_inf %arg : tensor<f32> -> tensor<i1>
  %2 = chlo.is_pos_inf %arg : tensor<f32> -> tensor<i1>
  // CHECK:      types0 = tensor<i1>
  // CHECK-NEXT: types0 = tensor<i1>
  // CHECK-NEXT: types0 = tensor<i1>
  %3 = "hlo_test_infer.get_return_types"(%0) : (tensor<i1>) -> tensor<i1>
  %4 = "hlo_test_infer.get_return_types"(%1) : (tensor<i1>) -> tensor<i1>
  %5 = "hlo_test_infer.get_return_types"(%2) : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}

// -----
// CHECK-LABEL: @infer_type_components_from_operands
func.func @infer_type_components_from_operands(%arg : tensor<2xf32>) -> tensor<2xf32> {
  %0 = "chlo.acos"(%arg) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "chlo.acosh"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  %2 = "chlo.asin"(%1) : (tensor<2xf32>) -> tensor<2xf32>
  %3 = "chlo.asinh"(%2) : (tensor<2xf32>) -> tensor<2xf32>
  %4 = "chlo.atan"(%3) : (tensor<2xf32>) -> tensor<2xf32>
  %5 = "chlo.atanh"(%4) : (tensor<2xf32>) -> tensor<2xf32>
  %6 = "chlo.bessel_i1e"(%5) : (tensor<2xf32>) -> tensor<2xf32>
  %7 = "chlo.conj"(%6) : (tensor<2xf32>) -> tensor<2xf32>
  %8 = "chlo.cosh"(%7) : (tensor<2xf32>) -> tensor<2xf32>
  %9 = "chlo.digamma"(%8) : (tensor<2xf32>) -> tensor<2xf32>
  %10 = "chlo.erf"(%9) : (tensor<2xf32>) -> tensor<2xf32>
  %11 = "chlo.erfc"(%10) : (tensor<2xf32>) -> tensor<2xf32>
  %12 = "chlo.lgamma"(%11) : (tensor<2xf32>) -> tensor<2xf32>
  %13 = "chlo.next_after"(%12, %12) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %14 = "chlo.polygamma"(%13, %13) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %15 = "chlo.sinh"(%14) : (tensor<2xf32>) -> tensor<2xf32>
  %16 = "chlo.tan"(%15) : (tensor<2xf32>) -> tensor<2xf32>
  %17 = "chlo.zeta"(%16, %16) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  // CHECK-NEXT: types0 = tensor<2xf32>
  %r0 = "hlo_test_infer.get_return_types"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  %r1 = "hlo_test_infer.get_return_types"(%1) : (tensor<2xf32>) -> tensor<2xf32>
  %r2 = "hlo_test_infer.get_return_types"(%2) : (tensor<2xf32>) -> tensor<2xf32>
  %r3 = "hlo_test_infer.get_return_types"(%3) : (tensor<2xf32>) -> tensor<2xf32>
  %r4 = "hlo_test_infer.get_return_types"(%4) : (tensor<2xf32>) -> tensor<2xf32>
  %r5 = "hlo_test_infer.get_return_types"(%5) : (tensor<2xf32>) -> tensor<2xf32>
  %r6 = "hlo_test_infer.get_return_types"(%6) : (tensor<2xf32>) -> tensor<2xf32>
  %r7 = "hlo_test_infer.get_return_types"(%7) : (tensor<2xf32>) -> tensor<2xf32>
  %r8 = "hlo_test_infer.get_return_types"(%8) : (tensor<2xf32>) -> tensor<2xf32>
  %r9 = "hlo_test_infer.get_return_types"(%9) : (tensor<2xf32>) -> tensor<2xf32>
  %r10 = "hlo_test_infer.get_return_types"(%10) : (tensor<2xf32>) -> tensor<2xf32>
  %r11 = "hlo_test_infer.get_return_types"(%11) : (tensor<2xf32>) -> tensor<2xf32>
  %r12 = "hlo_test_infer.get_return_types"(%12) : (tensor<2xf32>) -> tensor<2xf32>
  %r13 = "hlo_test_infer.get_return_types"(%13) : (tensor<2xf32>) -> tensor<2xf32>
  %r14 = "hlo_test_infer.get_return_types"(%14) : (tensor<2xf32>) -> tensor<2xf32>
  %r15 = "hlo_test_infer.get_return_types"(%15) : (tensor<2xf32>) -> tensor<2xf32>
  %r16 = "hlo_test_infer.get_return_types"(%16) : (tensor<2xf32>) -> tensor<2xf32>
  %r17 = "hlo_test_infer.get_return_types"(%17) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %r17 : tensor<2xf32>
}
