// RUN: stablehlo-opt --stablehlo-aggressive-folder=fold-op-element-limit=100 --split-input-file --verify-diagnostics %s | FileCheck %s

////////
// AddOp

// CHECK-LABEL: @add_fold_cst
func.func @add_fold_cst() -> (tensor<i32>, tensor<f32>) {
  %cst = stablehlo.constant dense<1> : tensor<i32>
  %cst_1 = stablehlo.constant dense<1.0> : tensor<f32>
  // CHECK: stablehlo.constant dense<2> : tensor<i32>
  // CHECK: stablehlo.constant dense<2.0{{.*}}> : tensor<f32>
  %0 = stablehlo.add %cst, %cst : tensor<i32>
  %1 = stablehlo.add %cst_1, %cst_1 : tensor<f32>
  return %0, %1 : tensor<i32>, tensor<f32>
}

// -----

////////
// BroadcastInDimOp

// CHECK-LABEL: func.func @broadcast_in_dim_fold_splat
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<3x3xi32>)
func.func @broadcast_in_dim_fold_splat(%arg0: tensor<3x3xi32>)
  -> (tensor<6xi32>, tensor<3xf32>, tensor<5xcomplex<f32>>, tensor<3x3xi32>) {
  %c0 = stablehlo.constant dense<5> : tensor<i32>
  %c1 = stablehlo.constant dense<3.0> : tensor<f32>
  %c2 = stablehlo.constant dense<(1.0,2.0)> : tensor<complex<f32>>
  %c3 = stablehlo.constant dense<1> : tensor<1x3xi32>

  %0 = stablehlo.broadcast_in_dim %c0, dims = [] : (tensor<i32>) -> tensor<6xi32>
  %1 = stablehlo.broadcast_in_dim %c1, dims = [] : (tensor<f32>) -> tensor<3xf32>
  %2 = stablehlo.broadcast_in_dim %c2, dims = [] : (tensor<complex<f32>>) -> tensor<5xcomplex<f32>>
  %3 = stablehlo.broadcast_in_dim %c3, dims = [1, 0] : (tensor<1x3xi32>) -> tensor<3x3xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.constant dense<5> : tensor<6xi32>
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.constant dense<3.000000e+00> : tensor<3xf32>
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.constant dense<(1.0{{.*}},2.0{{.*}})> : tensor<5xcomplex<f32>>
  // CHECK-DAG:  [[R3:%.+]] = stablehlo.constant dense<1> : tensor<3x3xi32>

  // CHECK-NEXT: return [[R0]], [[R1]], [[R2]], [[R3]]
  return %0, %1, %2, %3 : tensor<6xi32>, tensor<3xf32>, tensor<5xcomplex<f32>>, tensor<3x3xi32>
}

// -----

////////
// CaseOp

// CHECK-LABEL: func.func @case_fold_constant_branch_index_int_result
func.func @case_fold_constant_branch_index_int_result(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  // CHECK-NEXT: {{(^ *|func\.)}}return %arg1
  // CHECK-NOT:  stablehlo.case
  %branch_index = stablehlo.constant dense<1> : tensor<i32>
  %result = "stablehlo.case"(%branch_index) ({
    stablehlo.return %arg0 : tensor<i32>
  }, {
    stablehlo.return %arg1 : tensor<i32>
  }, {
    stablehlo.return %arg2 : tensor<i32>
  }) : (tensor<i32>) -> tensor<i32>
  func.return %result: tensor<i32>
}

// -----

// CHECK-LABEL: func.func @case_fold_constant_branch_index_complex_result
func.func @case_fold_constant_branch_index_complex_result(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>, %arg2: tensor<complex<f32>>) -> tensor<complex<f32>> {
  // CHECK-NEXT: {{(^ *|func\.)}}return %arg1
  // CHECK-NOT:  stablehlo.case
  %branch_index = stablehlo.constant dense<1> : tensor<i32>
  %result = "stablehlo.case"(%branch_index) ({
    stablehlo.return %arg0 : tensor<complex<f32>>
  }, {
    stablehlo.return %arg1 : tensor<complex<f32>>
  }, {
    stablehlo.return %arg2 : tensor<complex<f32>>
  }) : (tensor<i32>) -> tensor<complex<f32>>
  func.return %result: tensor<complex<f32>>
}

// -----

// CHECK-LABEL: func.func @case_fold_inline_call_tf_function
func.func @case_fold_inline_call_tf_function(%arg0: !stablehlo.token {jax.token = true}, %arg1: tensor<16xi32>, %arg2: tensor<16xi64>) -> (!stablehlo.token {jax.token = true}, tensor<16xi32> {jax.result_info = "result"}) {
  // CHECK: [[RESULT_TOKEN:%.+]] = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1, %arg2)
  // CHECK: [[UNUSED_TOKEN:%.+]] = {{"?}}stablehlo.case{{"?}}(
  // CHECK: return [[RESULT_TOKEN]], %arg1
  %c = stablehlo.constant dense<1> : tensor<i32>
  %c_0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.case"(%c_0) ({
    stablehlo.return %c_0 : tensor<i32>
  }, {
    stablehlo.return %c : tensor<i32>
  }) : (tensor<i32>) -> tensor<i32>
  %1 = "stablehlo.case"(%0) ({
    %2 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1, %arg2) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {called_index = 0 : i64, has_token_input_output = true}} : (!stablehlo.token, tensor<16xi32>, tensor<16xi64>) -> !stablehlo.token
    stablehlo.return %2 : !stablehlo.token
  }, {
    %2 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1, %arg2) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {called_index = 1 : i64, has_token_input_output = true}} : (!stablehlo.token, tensor<16xi32>, tensor<16xi64>) -> !stablehlo.token
    stablehlo.return %2 : !stablehlo.token
  }) : (tensor<i32>) -> !stablehlo.token
  return %1, %arg1 : !stablehlo.token, tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @case_fold_preserve_side_effects
func.func @case_fold_preserve_side_effects(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<3x2xi32>, tensor<f32>) {
  // COM:       // Inline the executed branch of the `case` op:
  // CHECK-DAG: [[BAR:%.+]] = stablehlo.custom_call @bar(%arg1) {has_side_effect = true}

  // COM:       // Replace the inlined branch with a trivial placeholder that
  // COM:       // just returns arbitrary constants matching the original type
  // COM:       // signature.
  // CHECK-DAG: [[PLACEHOLDER_INT_TENSOR:%.+]] = stablehlo.constant dense<0> : tensor<3x2xi32>
  // CHECK-DAG: [[PLACEHOLDER_FLOAT:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // COM:       // Keep the rest of the `case` op if it's non-trivially dead:
  // CHECK-DAG: [[BRANCH_INDEX:%.+]] = stablehlo.constant dense<1> : tensor<i32>

  // CHECK-DAG: [[NON_TRIVIALLY_DEAD_CASE_OP:%.+]] = {{"?}}stablehlo.case{{"?}}([[BRANCH_INDEX]]) ({
  // COM:         // Non-trivially dead branches are preserved but unused.
  // CHECK-DAG:   [[FOO:%.+]] = stablehlo.custom_call @foo(%arg0) {has_side_effect = true}
  // CHECK-DAG:   stablehlo.return [[FOO]], %arg0
  // COM:       }, {
  // COM:         // The executed branch is now just a trivial placeholder; its
  // COM:         // original logic has been inlined outside of the `case` op,
  // COM:         // and the replacement logic just returns arbitrary constants
  // COM:         // matching the original type signature.
  // CHECK-DAG:   stablehlo.return [[PLACEHOLDER_INT_TENSOR]], [[PLACEHOLDER_FLOAT]]
  // COM:       }, {
  // COM:         // Non-trivially dead branches are preserved but unused.
  // CHECK-DAG:   [[BAZ:%.+]] = stablehlo.custom_call @baz(%arg2) {has_side_effect = true}
  // CHECK-DAG:   stablehlo.return [[BAZ]], %arg2
  // COM:       })

  // COM:       // Return the result of the inlined branch.
  // CHECK-DAG: {{(^ *|func\.)}}return [[BAR]], %arg1
  %branch_index = stablehlo.constant dense<1> : tensor<i32>
  %result:2 = "stablehlo.case"(%branch_index) ({
    %foo = stablehlo.custom_call @foo(%arg0) {has_side_effect = true} : (tensor<f32>) -> tensor<3x2xi32>
    stablehlo.return %foo, %arg0 : tensor<3x2xi32>, tensor<f32>
  }, {
    %bar = stablehlo.custom_call @bar(%arg1) {has_side_effect = true} : (tensor<f32>) -> tensor<3x2xi32>
    stablehlo.return %bar, %arg1 : tensor<3x2xi32>, tensor<f32>
  }, {
    %baz = stablehlo.custom_call @baz(%arg2) {has_side_effect = true} : (tensor<f32>) -> tensor<3x2xi32>
    stablehlo.return %baz, %arg2 : tensor<3x2xi32>, tensor<f32>
  }) : (tensor<i32>) -> (tensor<3x2xi32>, tensor<f32>)
  func.return %result#0, %result#1 : tensor<3x2xi32>, tensor<f32>
}

// -----

// TODO: Allow non-trivially dead `case` ops to be simplified.

// CHECK-LABEL: func.func @case_fold_non_trivially_dead
func.func @case_fold_non_trivially_dead(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  // DISABLED-CHECK-NEXT: [[UNUSED:%.+]] = stablehlo.custom_call @bar(%arg1) {has_side_effect = true}
  // DISABLED-CHECK-NEXT: {{(^ *|func\.)}}return %arg1
  // DISABLED-CHECK-NOT:  stablehlo.case
  %branch_index = stablehlo.constant dense<1> : tensor<i32>
  %unused_case = "stablehlo.case"(%branch_index) ({
    %unused_foo = stablehlo.custom_call @foo(%arg0) {has_side_effect = false} : (tensor<i32>) -> tensor<i32>
    stablehlo.return %arg0 : tensor<i32>
  }, {
    %unused_bar = stablehlo.custom_call @bar(%arg1) {has_side_effect = true} : (tensor<i32>) -> tensor<i32>
    stablehlo.return %arg1 : tensor<i32>
  }, {
    %unused_baz = stablehlo.custom_call @baz(%arg2) {has_side_effect = false} : (tensor<i32>) -> tensor<i32>
    stablehlo.return %arg2 : tensor<i32>
  }) : (tensor<i32>) -> tensor<i32>
  func.return %arg1: tensor<i32>
}

// -----

////////
// ClampOp

// CHECK-LABEL: func.func @clamp_fold
func.func @clamp_fold(%arg0: tensor<3xi32>) -> tensor<3xi32> {
  %min = stablehlo.constant dense<[1, 5, 10]> : tensor<3xi32>
  %max = stablehlo.constant dense<[10, 25, 12]> : tensor<3xi32>
  %operand = stablehlo.constant dense<[0, 30, 11]> : tensor<3xi32>
  // CHECK: stablehlo.constant dense<[1, 25, 11]> : tensor<3xi32>
  %0 = "stablehlo.clamp"(%min, %operand, %max) : (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  func.return %0: tensor<3xi32>
}

// -----

////////
// CompareOp

// CHECK-LABEL: func.func @compare_fold_int
func.func @compare_fold_int()
  -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %cn1 = stablehlo.constant dense<-1> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c5 = stablehlo.constant dense<5> : tensor<i32>

  %0 = stablehlo.compare EQ, %cn1, %cn1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = stablehlo.compare GT, %c5, %c5, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare GE, %c4, %cn1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare LE, %c4, %c5, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %4 = stablehlo.compare EQ, %cn1, %cn1, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %5 = stablehlo.compare GT, %c5, %cn1, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %6 = stablehlo.compare GE, %c5, %c4, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %7 = stablehlo.compare LE, %cn1, %c5, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  // CHECK-DAG:  [[FALSE:%.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  [[TRUE:%.+]] = stablehlo.constant dense<true> : tensor<i1>

  // CHECK-NEXT: return [[TRUE]], [[FALSE]], [[TRUE]], [[TRUE]], [[TRUE]], [[FALSE]], [[TRUE]], [[FALSE]]
  return %0, %1, %2, %3, %4, %5, %6, %7 :
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// -----

// CHECK-LABEL: func.func @compare_fold_with_implicit_comparison_type
func.func @compare_fold_with_implicit_comparison_type() -> (tensor<3xi1>, tensor<3xi1>) {
  %c_0 = stablehlo.constant dense<0> : tensor<3xi64>
  %c = stablehlo.constant dense<[-1, 0, 1]> : tensor<3xi64>
  %c_1 = stablehlo.constant dense<0.0> : tensor<3xf64>
  %c_2 = stablehlo.constant dense<[-1.0, 0.0, 1.0]> : tensor<3xf64>
  %0 = stablehlo.compare GE, %c, %c_0 : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
  %1 = stablehlo.compare GE, %c_2, %c_1 : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
  // CHECK-DAG:  [[RES:%.+]] = stablehlo.constant dense<[false, true, true]> : tensor<3xi1>
  // CHECK-NEXT: return [[RES]], [[RES]] : tensor<3xi1>, tensor<3xi1>
  return %0, %1 : tensor<3xi1>, tensor<3xi1>
}

// -----

// CHECK-LABEL: func.func @compare_fold_float
func.func @compare_fold_float()
  -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %c1 = stablehlo.constant dense<0.01> : tensor<f32>
  %c2 = stablehlo.constant dense<-0.01> : tensor<f32>
  %c3 = stablehlo.constant dense<42.1> : tensor<f32>
  %c4 = stablehlo.constant dense<-50.0> : tensor<f32>

  %0 = stablehlo.compare EQ, %c0, %c0, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %1 = stablehlo.compare EQ, %c1, %c2, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %2 = stablehlo.compare NE, %c0, %c0, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %3 = stablehlo.compare NE, %c1, %c2, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %4 = stablehlo.compare GT, %c3, %c3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %5 = stablehlo.compare GT, %c3, %c4, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %6 = stablehlo.compare GE, %c3, %c3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %7 = stablehlo.compare GE, %c3, %c4, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %8 = stablehlo.compare LT, %c2, %c2, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %9 = stablehlo.compare LT, %c2, %c4, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %10 = stablehlo.compare LE, %c2, %c2, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %11 = stablehlo.compare LE, %c2, %c4, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>

  // CHECK-DAG:  [[FALSE:%.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  [[TRUE:%.+]] = stablehlo.constant dense<true> : tensor<i1>

  // CHECK-NEXT: return [[TRUE]], [[FALSE]], [[FALSE]], [[TRUE]], [[FALSE]], [[TRUE]], [[TRUE]], [[TRUE]], [[FALSE]], [[FALSE]], [[TRUE]], [[FALSE]]
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11 :
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// -----

// CHECK-LABEL: func.func @compare_fold_float_edge_cases
func.func @compare_fold_float_edge_cases()
  -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,

      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,

      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,

      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,

      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,

      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
      tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %pos_inf = stablehlo.constant dense<0x7F800000> : tensor<f32>
  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %nan = stablehlo.constant dense<0x7FC00000> : tensor<f32>

  // CHECK-DAG:  [[FALSE:%.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  [[TRUE:%.+]] = stablehlo.constant dense<true> : tensor<i1>

  %0 = stablehlo.compare EQ, %zero, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %1 = stablehlo.compare EQ, %zero, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %2 = stablehlo.compare EQ, %zero, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %3 = stablehlo.compare EQ, %zero, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %4 = stablehlo.compare EQ, %pos_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %5 = stablehlo.compare EQ, %pos_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %6 = stablehlo.compare EQ, %pos_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %7 = stablehlo.compare EQ, %pos_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %8 = stablehlo.compare EQ, %neg_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %9 = stablehlo.compare EQ, %neg_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %10 = stablehlo.compare EQ, %neg_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %11 = stablehlo.compare EQ, %neg_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %12 = stablehlo.compare EQ, %nan, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %13 = stablehlo.compare EQ, %nan, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %14 = stablehlo.compare EQ, %nan, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %15 = stablehlo.compare EQ, %nan, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>

  %16 = stablehlo.compare NE, %zero, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %17 = stablehlo.compare NE, %zero, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %18 = stablehlo.compare NE, %zero, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %19 = stablehlo.compare NE, %zero, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %20 = stablehlo.compare NE, %pos_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %21 = stablehlo.compare NE, %pos_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %22 = stablehlo.compare NE, %pos_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %23 = stablehlo.compare NE, %pos_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %24 = stablehlo.compare NE, %neg_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %25 = stablehlo.compare NE, %neg_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %26 = stablehlo.compare NE, %neg_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %27 = stablehlo.compare NE, %neg_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %28 = stablehlo.compare NE, %nan, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %29 = stablehlo.compare NE, %nan, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %30 = stablehlo.compare NE, %nan, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %31 = stablehlo.compare NE, %nan, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>

  %32 = stablehlo.compare GT, %zero, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %33 = stablehlo.compare GT, %zero, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %34 = stablehlo.compare GT, %zero, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %35 = stablehlo.compare GT, %zero, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %36 = stablehlo.compare GT, %pos_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %37 = stablehlo.compare GT, %pos_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %38 = stablehlo.compare GT, %pos_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %39 = stablehlo.compare GT, %pos_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %40 = stablehlo.compare GT, %neg_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %41 = stablehlo.compare GT, %neg_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %42 = stablehlo.compare GT, %neg_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %43 = stablehlo.compare GT, %neg_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %44 = stablehlo.compare GT, %nan, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %45 = stablehlo.compare GT, %nan, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %46 = stablehlo.compare GT, %nan, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %47 = stablehlo.compare GT, %nan, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>

  %48 = stablehlo.compare GE, %zero, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %49 = stablehlo.compare GE, %zero, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %50 = stablehlo.compare GE, %zero, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %51 = stablehlo.compare GE, %zero, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %52 = stablehlo.compare GE, %pos_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %53 = stablehlo.compare GE, %pos_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %54 = stablehlo.compare GE, %pos_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %55 = stablehlo.compare GE, %pos_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %56 = stablehlo.compare GE, %neg_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %57 = stablehlo.compare GE, %neg_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %58 = stablehlo.compare GE, %neg_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %59 = stablehlo.compare GE, %neg_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %60 = stablehlo.compare GE, %nan, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %61 = stablehlo.compare GE, %nan, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %62 = stablehlo.compare GE, %nan, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %63 = stablehlo.compare GE, %nan, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>

  %64 = stablehlo.compare LT, %zero, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %65 = stablehlo.compare LT, %zero, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %66 = stablehlo.compare LT, %zero, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %67 = stablehlo.compare LT, %zero, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %68 = stablehlo.compare LT, %pos_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %69 = stablehlo.compare LT, %pos_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %70 = stablehlo.compare LT, %pos_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %71 = stablehlo.compare LT, %pos_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %72 = stablehlo.compare LT, %neg_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %73 = stablehlo.compare LT, %neg_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %74 = stablehlo.compare LT, %neg_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %75 = stablehlo.compare LT, %neg_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %76 = stablehlo.compare LT, %nan, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %77 = stablehlo.compare LT, %nan, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %78 = stablehlo.compare LT, %nan, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %79 = stablehlo.compare LT, %nan, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>

  %80 = stablehlo.compare LE, %zero, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %81 = stablehlo.compare LE, %zero, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %82 = stablehlo.compare LE, %zero, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %83 = stablehlo.compare LE, %zero, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %84 = stablehlo.compare LE, %pos_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %85 = stablehlo.compare LE, %pos_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %86 = stablehlo.compare LE, %pos_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %87 = stablehlo.compare LE, %pos_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %88 = stablehlo.compare LE, %neg_inf, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %89 = stablehlo.compare LE, %neg_inf, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %90 = stablehlo.compare LE, %neg_inf, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %91 = stablehlo.compare LE, %neg_inf, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %92 = stablehlo.compare LE, %nan, %zero, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %93 = stablehlo.compare LE, %nan, %pos_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %94 = stablehlo.compare LE, %nan, %neg_inf, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %95 = stablehlo.compare LE, %nan, %nan, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>

  // CHECK: return [[TRUE]],  [[FALSE]], [[FALSE]], [[FALSE]],
  // CHECK-SAME:   [[FALSE]], [[TRUE]],  [[FALSE]], [[FALSE]],
  // CHECK-SAME:   [[FALSE]], [[FALSE]], [[TRUE]],  [[FALSE]],
  // CHECK-SAME:   [[FALSE]], [[FALSE]], [[FALSE]], [[FALSE]],

  // CHECK-SAME:   [[FALSE]], [[TRUE]],  [[TRUE]],  [[TRUE]],
  // CHECK-SAME:   [[TRUE]],  [[FALSE]], [[TRUE]],  [[TRUE]],
  // CHECK-SAME:   [[TRUE]],  [[TRUE]],  [[FALSE]], [[TRUE]],
  // CHECK-SAME:   [[TRUE]],  [[TRUE]],  [[TRUE]],  [[TRUE]],

  // CHECK-SAME:   [[FALSE]], [[FALSE]], [[TRUE]],  [[FALSE]],
  // CHECK-SAME:   [[TRUE]],  [[FALSE]], [[TRUE]],  [[FALSE]],
  // CHECK-SAME:   [[FALSE]], [[FALSE]], [[FALSE]], [[FALSE]],
  // CHECK-SAME:   [[FALSE]], [[FALSE]], [[FALSE]], [[FALSE]],

  // CHECK-SAME:   [[TRUE]],  [[FALSE]], [[TRUE]],  [[FALSE]],
  // CHECK-SAME:   [[TRUE]],  [[TRUE]],  [[TRUE]],  [[FALSE]],
  // CHECK-SAME:   [[FALSE]], [[FALSE]], [[TRUE]],  [[FALSE]],
  // CHECK-SAME:   [[FALSE]], [[FALSE]], [[FALSE]], [[FALSE]],

  // CHECK-SAME:   [[FALSE]], [[TRUE]],  [[FALSE]], [[FALSE]],
  // CHECK-SAME:   [[FALSE]], [[FALSE]], [[FALSE]], [[FALSE]],
  // CHECK-SAME:   [[TRUE]],  [[TRUE]],  [[FALSE]], [[FALSE]],
  // CHECK-SAME:   [[FALSE]], [[FALSE]], [[FALSE]], [[FALSE]],

  // CHECK-SAME:   [[TRUE]],  [[TRUE]],  [[FALSE]], [[FALSE]],
  // CHECK-SAME:   [[FALSE]], [[TRUE]],  [[FALSE]], [[FALSE]],
  // CHECK-SAME:   [[TRUE]],  [[TRUE]],  [[TRUE]],  [[FALSE]],
  // CHECK-SAME:   [[FALSE]], [[FALSE]], [[FALSE]], [[FALSE]]

  return  %0,  %1,  %2,  %3,
          %4,  %5,  %6,  %7,
          %8,  %9, %10, %11,
         %12, %13, %14, %15,

         %16, %17, %18, %19,
         %20, %21, %22, %23,
         %24, %25, %26, %27,
         %28, %29, %30, %31,

         %32, %33, %34, %35,
         %36, %37, %38, %39,
         %40, %41, %42, %43,
         %44, %45, %46, %47,

         %48, %49, %50, %51,
         %52, %53, %54, %55,
         %56, %57, %58, %59,
         %60, %61, %62, %63,

         %64, %65, %66, %67,
         %68, %69, %70, %71,
         %72, %73, %74, %75,
         %76, %77, %78, %79,

         %80, %81, %82, %83,
         %84, %85, %86, %87,
         %88, %89, %90, %91,
         %92, %93, %94, %95 :
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,

         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,

         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,

         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,

         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,

         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>,
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// -----

////////
// ConcatenateOp

// CHECK-LABEL: func.func @concatenate_fold
func.func @concatenate_fold() -> (tensor<6xi32>, tensor<3xi32>, tensor<3x3xi32>, tensor<2x5xi32>) {
  %c0 = stablehlo.constant dense<[0, 1]> : tensor<2xi32>
  %c1 = stablehlo.constant dense<[2, 3, 4]> : tensor<3xi32>
  %c2 = stablehlo.constant dense<[5]> : tensor<1xi32>

  %c3 = stablehlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %c4 = stablehlo.constant dense<[[6, 7, 8]]> : tensor<1x3xi32>
  %c5 = stablehlo.constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>

  %0 = stablehlo.concatenate %c0, %c1, %c2, dim = 0 : (tensor<2xi32>, tensor<3xi32>, tensor<1xi32>) -> tensor<6xi32>
  %1 = stablehlo.concatenate %c0, %c2, dim = 0 : (tensor<2xi32>, tensor<1xi32>) -> tensor<3xi32>

  %2 = stablehlo.concatenate %c3, %c4, dim = 0 : (tensor<2x3xi32>, tensor<1x3xi32>) -> tensor<3x3xi32>
  %3 = stablehlo.concatenate %c3, %c5, dim = 1 : (tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x5xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.constant dense<[0, 1, 2, 3, 4, 5]> : tensor<6xi32>
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.constant dense<[0, 1, 5]> : tensor<3xi32>
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.constant dense<{{\[\[0, 1, 2\], \[3, 4, 5\], \[6, 7, 8\]\]}}> : tensor<3x3xi32>
  // CHECK-DAG:  [[R3:%.+]] = stablehlo.constant dense<{{\[\[0, 1, 2, 11, 12\], \[3, 4, 5, 13, 14\]\]}}> : tensor<2x5xi32>
  // CHECK-NEXT: return [[R0]], [[R1]], [[R2]], [[R3]]
  return %0, %1, %2, %3 : tensor<6xi32>, tensor<3xi32>, tensor<3x3xi32>, tensor<2x5xi32>
}

// CHECK-LABEL: func.func @fold_concatenate_splat_leading
func.func @fold_concatenate_splat_leading(%arg0: tensor<1xi32>) -> tensor<3xi32> {
  // CHECK: [[CST0:%.+]] = stablehlo.constant dense<0> : tensor<2xi32>
  // CHECK-NEXT: stablehlo.concatenate [[CST0]], %arg0, dim = 0
  %cst0 = stablehlo.constant dense<0> : tensor<1xi32>
  %0 = stablehlo.concatenate %cst0, %cst0, %arg0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// CHECK-LABEL: func.func @fold_concatenate_splat_trailing
func.func @fold_concatenate_splat_trailing(%arg0: tensor<2xi32>) -> tensor<6xi32> {
  // CHECK: [[CST0:%.+]] = stablehlo.constant dense<0> : tensor<4xi32>
  // CHECK-NEXT: stablehlo.concatenate %arg0, [[CST0]], dim = 0
  %cst0 = stablehlo.constant dense<0> : tensor<2xi32>
  %0 = stablehlo.concatenate %arg0, %cst0, %cst0, dim = 0 : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<6xi32>
  return %0 : tensor<6xi32>
}

// CHECK-LABEL: func.func @fold_concatenate_splat_middle
func.func @fold_concatenate_splat_middle(%arg0: tensor<1xi32>) -> tensor<4xi32> {
  // CHECK: [[CST0:%.+]] = stablehlo.constant dense<0> : tensor<2xi32>
  // CHECK-NEXT: stablehlo.concatenate %arg0, [[CST0]], %arg0, dim = 0
  %cst0 = stablehlo.constant dense<0> : tensor<1xi32>
  %0 = stablehlo.concatenate %arg0, %cst0, %cst0, %arg0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func.func @fold_concatenate_splat_multiple
func.func @fold_concatenate_splat_multiple(%arg0: tensor<1xi32>) -> tensor<5xi32> {
  // CHECK-DAG: [[CST0:%.+]] = stablehlo.constant dense<0> : tensor<2xi32>
  // CHECK-DAG: [[CST1:%.+]] = stablehlo.constant dense<1> : tensor<2xi32>
  // CHECK-NEXT: stablehlo.concatenate [[CST0]], [[CST1]], %arg0, dim = 0
  %cst0 = stablehlo.constant dense<0> : tensor<1xi32>
  %cst1 = stablehlo.constant dense<1> : tensor<1xi32>
  %0 = stablehlo.concatenate %cst0, %cst0, %cst1, %cst1, %arg0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<5xi32>
  return %0 : tensor<5xi32>
}

// -----

////////
// DivOp

// CHECK-LABEL: @div_fold_cst
func.func @div_fold_cst() -> (tensor<i32>, tensor<ui32>, tensor<f32>) {
  %cst = stablehlo.constant dense<2> : tensor<i32>
  %cst_1 = stablehlo.constant dense<2> : tensor<ui32>
  %cst_2 = stablehlo.constant dense<2.0> : tensor<f32>
  // CHECK: stablehlo.constant dense<1> : tensor<i32>
  // CHECK: stablehlo.constant dense<1> : tensor<ui32>
  // CHECK: stablehlo.constant dense<1.0{{.*}}> : tensor<f32>
  %0 = stablehlo.divide %cst, %cst : tensor<i32>
  %1 = stablehlo.divide %cst_1, %cst_1 : tensor<ui32>
  %2 = stablehlo.divide %cst_2, %cst_2 : tensor<f32>
  return %0, %1, %2 : tensor<i32>, tensor<ui32>, tensor<f32>
}

// CHECK-LABEL: @div_fold_cst_zero_nan
func.func @div_fold_cst_zero_nan() -> (tensor<f32>) {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.divide %cst, %cst_0 : tensor<f32>
  // CHECK: stablehlo.constant dense<0x7F800000> : tensor<f32>
  // CHECK-NOT: stablehlo.divide
  return %0 : tensor<f32>
}

// CHECK-LABEL: @div_fold_cst_nan
func.func @div_fold_cst_nan() -> (tensor<f32>) {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
  %0 = stablehlo.divide %cst, %cst_0 : tensor<f32>
  // CHECK: stablehlo.constant dense<0x7FC00000> : tensor<f32>
  // CHECK-NOT: stablehlo.divide
  return %0 : tensor<f32>
}

// -----

////////
// MaximumOp

// CHECK-LABEL: @max_fold_cst_nan
func.func @max_fold_cst_nan() -> (tensor<f32>) {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
  %0 = stablehlo.maximum %cst, %cst_0 : tensor<f32>
  // CHECK: stablehlo.constant dense<0x7FC00000> : tensor<f32>
  // CHECK-NOT: stablehlo.maximum
  return %0 : tensor<f32>
}

// -----

////////
// MinimumOp

// CHECK-LABEL: @min_fold_cst_nan
func.func @min_fold_cst_nan() -> (tensor<f32>) {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
  %0 = stablehlo.minimum %cst, %cst_0 : tensor<f32>
  // CHECK: stablehlo.constant dense<0x7FC00000> : tensor<f32>
  // CHECK-NOT: stablehlo.minimum
  return %0 : tensor<f32>
}

// -----

////////
// MulOp

// CHECK-LABEL: @mul_fold_cst
func.func @mul_fold_cst() -> (tensor<i32>, tensor<f32>) {
  %cst = stablehlo.constant dense<2> : tensor<i32>
  %cst_1 = stablehlo.constant dense<2.0> : tensor<f32>
  // CHECK: stablehlo.constant dense<4> : tensor<i32>
  // CHECK: stablehlo.constant dense<4.0{{.*}}> : tensor<f32>
  %0 = stablehlo.multiply %cst, %cst : tensor<i32>
  %1 = stablehlo.multiply %cst_1, %cst_1 : tensor<f32>
  return %0, %1 : tensor<i32>, tensor<f32>
}

// -----

////////
// SubtractOp

// CHECK-LABEL: @subtract_fold_cst
func.func @subtract_fold_cst() -> (tensor<i32>, tensor<f32>) {
  %cst = stablehlo.constant dense<1> : tensor<i32>
  %cst_1 = stablehlo.constant dense<3> : tensor<i32>
  %cst_2 = stablehlo.constant dense<1.0> : tensor<f32>
  %cst_3 = stablehlo.constant dense<3.0> : tensor<f32>
  // CHECK: stablehlo.constant dense<2> : tensor<i32>
  // CHECK: stablehlo.constant dense<2.0{{.*}}> : tensor<f32>
  %0 = stablehlo.subtract %cst_1, %cst : tensor<i32>
  %1 = stablehlo.subtract %cst_3, %cst_2 : tensor<f32>
  return %0, %1 : tensor<i32>, tensor<f32>
}

// -----

////////
// IotaOp

// CHECK-LABEL: func @eval_iota
func.func @eval_iota() -> (tensor<1xi32>, tensor<3x4x5xi32>, tensor<3x4x5xi32>) {
  // CHECK:      [[RESULT0:%.*]] = stablehlo.constant dense<0> : tensor<1xi32>
  // CHECK-NEXT: [[RESULT1:%.*]] = stablehlo.iota dim = 1 : tensor<3x4x5xi32>
  // CHECK-NEXT: [[RESULT2:%.*]] = stablehlo.iota dim = 2 : tensor<3x4x5xi32>
  // CHECK: return [[RESULT0]], [[RESULT1]], [[RESULT2]]
  %0 = stablehlo.iota dim = 0 : tensor<1xi32>
  %1 = stablehlo.iota dim = 1 : tensor<3x4x5xi32>
  %2 = stablehlo.iota dim = 2 : tensor<3x4x5xi32>
  func.return %0, %1, %2 : tensor<1xi32>, tensor<3x4x5xi32>, tensor<3x4x5xi32>
}

// -----

// CHECK-LABEL: func @eval_iota_zero_dimension
func.func @eval_iota_zero_dimension() -> (tensor<0xi32>, tensor<5x0x2xi32>) {
  // CHECK-NOT: stablehlo.iota
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<> : tensor<0xi32>
  // CHECK: [[RESULT1:%.*]] = stablehlo.constant dense<> : tensor<5x0x2xi32>
  // CHECK: return [[RESULT0]], [[RESULT1]]
  %0 = stablehlo.iota dim = 0 : tensor<0xi32>
  %1 = stablehlo.iota dim = 2 : tensor<5x0x2xi32>
  func.return %0, %1 : tensor<0xi32>, tensor<5x0x2xi32>
}

// -----

////////
// ReduceOp

// CHECK-LABEL: func @reduce_op_fold
func.func @reduce_op_fold(%arg0: tensor<i64>) -> tensor<i1> {
  %c = stablehlo.constant dense<false> : tensor<4x32xi1>
  %c_0 = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-NOT: stablehlo.reduce
  %0 = stablehlo.reduce(%c init: %c_0) applies stablehlo.or across dimensions = [0, 1] : (tensor<4x32xi1>, tensor<i1>) -> tensor<i1>
  return %0 : tensor<i1>
}

// -----

////////
// ReshapeOp

// CHECK-LABEL: func @reshape_fold
func.func @reshape_fold() -> (tensor<1xf32>, tensor<2x2xi32>, tensor<3x2xcomplex<f32>>) {
  %c0 = stablehlo.constant dense<2.0> : tensor<f32>
  %c1 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %c2 = stablehlo.constant dense<(1.0,2.0)> : tensor<2x3xcomplex<f32>>
  %0 = stablehlo.reshape %c0 : (tensor<f32>) -> tensor<1xf32>
  %1 = stablehlo.reshape %c1 : (tensor<4xi32>) -> tensor<2x2xi32>
  %2 = stablehlo.reshape %c2 : (tensor<2x3xcomplex<f32>>) -> tensor<3x2xcomplex<f32>>

  // CHECK-DAG:  [[RESULT0:%.+]] = stablehlo.constant dense<2.0{{.*}}> : tensor<1xf32>
  // CHECK-DAG:  [[RESULT1:%.+]] = stablehlo.constant dense<{{\[\[1, 2\], \[3, 4\]\]}}> : tensor<2x2xi32>
  // CHECK-DAG:  [[RESULT2:%.+]] = stablehlo.constant dense<(1.0{{.*}},2.0{{.*}})> : tensor<3x2xcomplex<f32>>
  // CHECK-NEXT: return [[RESULT0]], [[RESULT1]], [[RESULT2]]
  return %0, %1, %2 : tensor<1xf32>, tensor<2x2xi32>, tensor<3x2xcomplex<f32>>
}

// -----

////////
// SliceOp / DynamicSliceOp

// CHECK-LABEL: @slice_fold
func.func @slice_fold(%arg0: tensor<6x1xi32>) -> tensor<1x1xi32> {
  %c = stablehlo.constant dense<[[0], [1], [2], [3], [4], [5]]> : tensor<6x1xi32>
  %0 = stablehlo.slice %c [2:3, 0:1] : (tensor<6x1xi32>) -> tensor<1x1xi32>
  // CHECK: stablehlo.constant dense<2> : tensor<1x1xi32>
  return %0 : tensor<1x1xi32>
}

// CHECK-LABEL: @slice_fold_splat
func.func @slice_fold_splat(%arg0: tensor<6x1xi32>) -> tensor<1x1xi32> {
  %c = stablehlo.constant dense<1> : tensor<6x1xi32>
  %0 = stablehlo.slice %c [2:3, 0:1] : (tensor<6x1xi32>) -> tensor<1x1xi32>
  // CHECK: stablehlo.constant dense<1> : tensor<1x1xi32>
  return %0 : tensor<1x1xi32>
}

// CHECK-LABEL: @dynamic_slice_fold
func.func @dynamic_slice_fold(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<1x1xi32> {
  %0 = stablehlo.constant dense<256> : tensor<6x1xi32>
  %1 = "stablehlo.dynamic_slice"(%0, %arg0, %arg1) <{slice_sizes = array<i64: 1, 1>}> : (tensor<6x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x1xi32>

  // CHECK: %[[RESULT:.*]] = stablehlo.constant dense<256> : tensor<1x1xi32>
  // CHECK: return %[[RESULT]]
  return %1 : tensor<1x1xi32>
}

// -----

////////
// ConvertOp

// CHECK-LABEL: func @eval_convert_f32_to_i64
func.func @eval_convert_f32_to_i64() -> tensor<2xi64> {
  // CHECK-NOT: stablehlo.convert
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %1 = stablehlo.convert %0 : (tensor<2xf32>) -> tensor<2xi64>
  func.return %1 : tensor<2xi64>
}

// CHECK-LABEL: func @eval_convert_bool_f32
func.func @eval_convert_bool_f32() -> tensor<2xf32> {
  // CHECK-NEXT: [[CST:%.+]] = stablehlo.constant dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>
  %cst = stablehlo.constant dense<[0, 1]> : tensor<2xi1>
  %0 = stablehlo.convert %cst : (tensor<2xi1>) -> tensor<2xf32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @eval_convert_bool_i32
func.func @eval_convert_bool_i32() -> tensor<2xi32> {
  // CHECK-NEXT: [[CST:%.+]] = stablehlo.constant dense<[0, 1]> : tensor<2xi32>
  %cst = stablehlo.constant dense<[0, 1]> : tensor<2xi1>
  %0 = stablehlo.convert %cst : (tensor<2xi1>) -> tensor<2xi32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<2xi32>
}

// CHECK-LABEL: func @eval_convert_i32_bool
func.func @eval_convert_i32_bool() -> tensor<3xi1> {
  // CHECK-NEXT: [[CST:%.+]] = stablehlo.constant dense<[false, true, true]> : tensor<3xi1>
  %cst = stablehlo.constant dense<[0, 1, 10]> : tensor<3xi32>
  %0 = stablehlo.convert %cst : (tensor<3xi32>) -> tensor<3xi1>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<3xi1>
}

// CHECK-LABEL: func @eval_convert_f32_bool
func.func @eval_convert_f32_bool() -> tensor<4xi1> {
  // CHECK-NEXT: [[CST:%.+]] = stablehlo.constant dense<[true, false, true, true]> : tensor<4xi1>
  %cst = stablehlo.constant dense<[-1.0, 0.0, 1.0, 10.0]> : tensor<4xf32>
  %0 = stablehlo.convert %cst : (tensor<4xf32>) -> tensor<4xi1>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: func @eval_convert_f32_non_convertable
func.func @eval_convert_f32_non_convertable() -> (tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.convert
  // CHECK: [[RESULT1:%.*]] = stablehlo.convert
  // CHECK: [[RESULT2:%.*]] = stablehlo.convert
  // CHECK: return [[RESULT0]], [[RESULT1]], [[RESULT2]]
  %pinf = stablehlo.constant dense<[1.0, 0x7F800000]> : tensor<2xf32>
  %ninf = stablehlo.constant dense<[2.0, 0xFF800000]> : tensor<2xf32>
  %nzero = stablehlo.constant dense<[3.0, 0x80000000]> : tensor<2xf32>
  %0 = stablehlo.convert %pinf : (tensor<2xf32>) -> tensor<2xi64>
  %1 = stablehlo.convert %ninf : (tensor<2xf32>) -> tensor<2xi64>
  %2 = stablehlo.convert %nzero : (tensor<2xf32>) -> tensor<2xi64>
  func.return %0, %1, %2 : tensor<2xi64>, tensor<2xi64>, tensor<2xi64>
}

// -----

// CHECK-LABEL: func @eval_convert_f32_non_fittable
func.func @eval_convert_f32_non_fittable() -> (tensor<1xi32>, tensor<1xi32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<2.14748365E+9> : tensor<1xf32>
  // CHECK: [[RESULT1:%.*]] = stablehlo.constant dense<2147483520> : tensor<1xi32>
  // CHECK: [[RESULT2:%.*]] = stablehlo.convert [[RESULT0]]
  // CHECK: return [[RESULT1]], [[RESULT2]]
  %twopow30 = stablehlo.constant dense<[2147483583.0]> : tensor<1xf32>
  %twopow31 = stablehlo.constant dense<[2147483584.0]> : tensor<1xf32>
  %1 = stablehlo.convert %twopow30 : (tensor<1xf32>) -> tensor<1xi32>
  %2 = stablehlo.convert %twopow31 : (tensor<1xf32>) -> tensor<1xi32>
  func.return %1, %2 : tensor<1xi32>, tensor<1xi32>
}

// -----

// CHECK-LABEL: func @eval_convert_i32_non_exact
func.func @eval_convert_i32_non_exact() -> (tensor<1xf32>, tensor<1xf32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<0x4B7FFFFF> : tensor<1xf32>
  // 0x4B800000 = 16777216, error due to conversion -1
  // CHECK: [[RESULT1:%.*]] = stablehlo.constant dense<0x4B800000> : tensor<1xf32>
  // CHECK: return [[RESULT0]], [[RESULT1]]
  %pow23 = stablehlo.constant dense<[16777215]> : tensor<1xi32>
  %pow24 = stablehlo.constant dense<[16777217]> : tensor<1xi32>
  %1 = stablehlo.convert %pow23 : (tensor<1xi32>) -> tensor<1xf32>
  %2 = stablehlo.convert %pow24 : (tensor<1xi32>) -> tensor<1xf32>
  func.return %1, %2 : tensor<1xf32>, tensor<1xf32>
}

// -----

// CHECK-LABEL: func @eval_convert_f64_precision_loss
func.func @eval_convert_f64_precision_loss() -> (tensor<1xf32>, tensor<f32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<9.99999996E-13> : tensor<1xf32>
  // CHECK: return [[RESULT0]]
  %0 = arith.constant dense<9.9999999999999998E-13> : tensor<1xf64>
  %1 = stablehlo.constant dense<8.000000e+00> : tensor<f64>
  %2 = stablehlo.convert %0 : (tensor<1xf64>) -> tensor<1xf32>
  %3 = stablehlo.convert %1 : (tensor<f64>) -> tensor<f32>
  func.return %2, %3 : tensor<1xf32>, tensor<f32>
}

// -----

////////
// AbsOp

// CHECK-LABEL: func @fold_abs
func.func @fold_abs() -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[INT_ZERO:%.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: [[INT_TEN:%.*]] = stablehlo.constant dense<10> : tensor<i32>
  // CHECK-DAG: [[FLOAT_ZERO:%.*]] = stablehlo.constant dense<0.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[FLOAT_HALF:%.*]] = stablehlo.constant dense<5.0{{.*}}e-01> : tensor<f32>
  // CHECK-DAG: [[FLOAT_INF:%.*]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
  // CHECK:     return [[INT_ZERO]], [[INT_TEN]], [[INT_TEN]], [[FLOAT_ZERO]], [[FLOAT_HALF]], [[FLOAT_HALF]], [[FLOAT_INF]], [[FLOAT_INF]]

  %int_zero = stablehlo.constant dense<0> : tensor<i32>
  %int_neg_ten = stablehlo.constant dense<-10> : tensor<i32>
  %int_pos_ten = stablehlo.constant dense<10> : tensor<i32>

  %float_zero = stablehlo.constant dense<0.0> : tensor<f32>
  %float_neg_half = stablehlo.constant dense<-0.5> : tensor<f32>
  %float_pos_half = stablehlo.constant dense<0.5> : tensor<f32>
  %float_neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32> // -inf
  %float_pos_inf = stablehlo.constant dense<0x7F800000> : tensor<f32> // +inf

  %0 = stablehlo.abs %int_zero : tensor<i32>
  %1 = stablehlo.abs %int_neg_ten : tensor<i32>
  %2 = stablehlo.abs %int_pos_ten : tensor<i32>

  %3 = stablehlo.abs %float_zero : tensor<f32>
  %4 = stablehlo.abs %float_neg_half : tensor<f32>
  %5 = stablehlo.abs %float_pos_half : tensor<f32>
  %6 = stablehlo.abs %float_neg_inf : tensor<f32>
  %7 = stablehlo.abs %float_pos_inf : tensor<f32>

  func.return %0, %1, %2, %3, %4, %5, %6, %7 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

////////
// CosineOp

// CHECK-LABEL: func @fold_cosine
func.func @fold_cosine() -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[ONE:%.*]] = stablehlo.constant dense<{{1\.0000.*}}> : tensor<f32>
  // CHECK-DAG: [[SQRT_THREE_OVER_TWO:%.*]] = stablehlo.constant dense<{{0\.8660.*|8\.660.*[Ee]-01}}> : tensor<f32>
  // CHECK-DAG: [[SQRT_TWO_OVER_TWO:%.*]] = stablehlo.constant dense<{{0\.7071.*|7\.071.*[Ee]-01}}> : tensor<f32>
  // CHECK-DAG: [[HALF:%.*]] = stablehlo.constant dense<{{0\.5000.*|5\.000.*[Ee]-01|0.4999.*|4\.999.*[Ee]-01}}> : tensor<f32>
  // CHECK-DAG: [[ZERO:%.*]] = stablehlo.constant dense<{{-?(0\.0000.*|[0-9]\.[0-9]*[Ee]-(0?[5-9]|[1-9][0-9]))}}> : tensor<f32>
  // CHECK:     return [[ONE]], [[SQRT_THREE_OVER_TWO]], [[SQRT_TWO_OVER_TWO]], [[HALF]], [[ZERO]]

  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.constant dense<0.5235987755982989> : tensor<f32> // pi/6
  %2 = stablehlo.constant dense<0.7853981633974483> : tensor<f32> // pi/4
  %3 = stablehlo.constant dense<1.0471975511965977> : tensor<f32> // pi/3
  %4 = stablehlo.constant dense<1.5707963267948966> : tensor<f32> // pi/2

  %5 = stablehlo.cosine %0 : tensor<f32>
  %6 = stablehlo.cosine %1 : tensor<f32>
  %7 = stablehlo.cosine %2 : tensor<f32>
  %8 = stablehlo.cosine %3 : tensor<f32>
  %9 = stablehlo.cosine %4 : tensor<f32>

  func.return %5, %6, %7, %8, %9 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

////////
// ErfOp

// CHECK-LABEL: func @fold_erf
func.func @fold_erf() -> (tensor<f32>, tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[RESULT0:%.*]] = stablehlo.constant dense<-0.52049{{.*}}> : tensor<f32>
  // CHECK-DAG: [[RESULT1:%.*]] = stablehlo.constant dense<0.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[RESULT2:%.*]] = stablehlo.constant dense<0.90000{{.*}}> : tensor<f32>
  // CHECK:     return [[RESULT0]], [[RESULT1]], [[RESULT2]]

  %0 = stablehlo.constant dense<-0.5> : tensor<f32>
  %1 = stablehlo.constant dense<0.0> : tensor<f32>
  %2 = stablehlo.constant dense<1.1631> : tensor<f32>

  %3 = chlo.erf %0 : tensor<f32> -> tensor<f32>
  %4 = chlo.erf %1 : tensor<f32> -> tensor<f32>
  %5 = chlo.erf %2 : tensor<f32> -> tensor<f32>

  func.return %3, %4, %5 : tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

////////
// ExpOp

// CHECK-LABEL: func @fold_exponential
func.func @fold_exponential() -> (tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[ONE:%.*]] = stablehlo.constant dense<1.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[E:%.*]] = stablehlo.constant dense<2.718{{.*}}> : tensor<f32>
  // CHECK:     return [[ONE]], [[E]]

  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.constant dense<1.0> : tensor<f32>

  %2 = stablehlo.exponential %0 : tensor<f32>
  %3 = stablehlo.exponential %1 : tensor<f32>

  func.return %2, %3 : tensor<f32>, tensor<f32>
}

// -----

////////
// LogOp

// CHECK-LABEL: func @fold_log
func.func @fold_log() -> (tensor<f32>, tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[ZERO:%.*]] = stablehlo.constant dense<0.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[ONE:%.*]] = stablehlo.constant dense<{{1\.0.*|0\.999.*}}> : tensor<f32>
  // CHECK-DAG: [[DO_NOT_FOLD_LOG_ZERO:%.*]] = stablehlo.log [[ZERO]] : tensor<f32>
  // CHECK:     return [[ZERO]], [[ONE]], [[DO_NOT_FOLD_LOG_ZERO]]

  %0 = stablehlo.constant dense<1.0> : tensor<f32>
  %1 = stablehlo.constant dense<2.718281828459045> : tensor<f32>
  %2 = stablehlo.constant dense<0.0> : tensor<f32>

  %3 = stablehlo.log %0 : tensor<f32>
  %4 = stablehlo.log %1 : tensor<f32>
  %5 = stablehlo.log %2 : tensor<f32>

  func.return %3, %4, %5 : tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

////////
// LogisticOp

// CHECK-LABEL: func @fold_logistic
func.func @fold_logistic() -> (tensor<f32>, tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[ZERO:%.*]] = stablehlo.constant dense<0.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[HALF:%.*]] = stablehlo.constant dense<5.0{{.*}}e-01> : tensor<f32>
  // CHECK-DAG: [[ONE:%.*]] = stablehlo.constant dense<1.0{{.*}}> : tensor<f32>
  // CHECK:     return [[ZERO]], [[HALF]], [[ONE]]

  %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %pos_inf = stablehlo.constant dense<0x7F800000> : tensor<f32>

  %0 = stablehlo.logistic %neg_inf : tensor<f32>
  %1 = stablehlo.logistic %zero : tensor<f32>
  %2 = stablehlo.logistic %pos_inf : tensor<f32>

  func.return %0, %1, %2 : tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

////////
// NegOp

// CHECK-LABEL: func @fold_negate
func.func @fold_negate() -> (tensor<i32>, tensor<i32>, tensor<f32>) {
  // CHECK-DAG: [[RESULT0:%.*]] = stablehlo.constant dense<-4> : tensor<i32>
  // CHECK-DAG: [[RESULT1:%.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: [[RESULT2:%.*]] = stablehlo.constant dense<9.999{{.*}}e+02> : tensor<f32>
  // CHECK:     return [[RESULT0]], [[RESULT1]], [[RESULT2]]

  %0 = stablehlo.constant dense<4> : tensor<i32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  %2 = stablehlo.constant dense<-999.9> : tensor<f32>

  %3 = stablehlo.negate %0 : tensor<i32>
  %4 = stablehlo.negate %1 : tensor<i32>
  %5 = stablehlo.negate %2 : tensor<f32>

  func.return %3, %4, %5 : tensor<i32>, tensor<i32>, tensor<f32>
}

// -----

////////
// NotOp

// CHECK-LABEL: func @fold_not
func.func @fold_not() -> (tensor<i32>, tensor<i32>) {
  // CHECK-DAG: [[RESULT0:%.*]] = stablehlo.constant dense<42> : tensor<i32>
  // CHECK-DAG: [[RESULT1:%.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // CHECK:     return [[RESULT0]], [[RESULT1]]

  %0 = stablehlo.constant dense<-43> : tensor<i32>
  %1 = stablehlo.constant dense<0> : tensor<i32>

  %2 = stablehlo.not %0 : tensor<i32>
  %3 = stablehlo.not %1 : tensor<i32>

  func.return %2, %3 : tensor<i32>, tensor<i32>
}

// -----

////////
// RoundOp

// CHECK-LABEL: func @fold_round_nearest_afz
func.func @fold_round_nearest_afz() -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[NEG_THREE:%.*]] = stablehlo.constant dense<-3.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[NEG_TWO:%.*]] = stablehlo.constant dense<-2.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[ZERO:%.*]] = stablehlo.constant dense<0.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[ONE:%.*]] = stablehlo.constant dense<1.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[TWO:%.*]] = stablehlo.constant dense<2.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[THREE:%.*]] = stablehlo.constant dense<3.0{{.*}}> : tensor<f32>
  // CHECK:     return [[NEG_THREE]], [[NEG_TWO]], [[ZERO]], [[ONE]], [[ONE]], [[TWO]], [[THREE]]

  %0 = stablehlo.constant dense<-2.5> : tensor<f32>
  %1 = stablehlo.constant dense<-1.5> : tensor<f32>
  %2 = stablehlo.constant dense<0.4> : tensor<f32>
  %3 = stablehlo.constant dense<0.5> : tensor<f32>
  %4 = stablehlo.constant dense<0.6> : tensor<f32>
  %5 = stablehlo.constant dense<1.5> : tensor<f32>
  %6 = stablehlo.constant dense<2.5> : tensor<f32>

  %7 = stablehlo.round_nearest_afz %0 : tensor<f32>
  %8 = stablehlo.round_nearest_afz %1 : tensor<f32>
  %9 = stablehlo.round_nearest_afz %2 : tensor<f32>
  %10 = stablehlo.round_nearest_afz %3 : tensor<f32>
  %11 = stablehlo.round_nearest_afz %4 : tensor<f32>
  %12 = stablehlo.round_nearest_afz %5 : tensor<f32>
  %13 = stablehlo.round_nearest_afz %6 : tensor<f32>

  func.return %7, %8, %9, %10, %11, %12, %13 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

////////
// RoundNearestEvenOp

// CHECK-LABEL: func @fold_round_nearest_even
func.func @fold_round_nearest_even() -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[NEG_TWO:%.*]] = stablehlo.constant dense<-2.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[ZERO:%.*]] = stablehlo.constant dense<0.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[ONE:%.*]] = stablehlo.constant dense<1.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[TWO:%.*]] = stablehlo.constant dense<2.0{{.*}}> : tensor<f32>
  // CHECK:     return [[NEG_TWO]], [[NEG_TWO]], [[ZERO]], [[ZERO]], [[ONE]], [[TWO]], [[TWO]]

  %0 = stablehlo.constant dense<-2.5> : tensor<f32>
  %1 = stablehlo.constant dense<-1.5> : tensor<f32>
  %2 = stablehlo.constant dense<0.4> : tensor<f32>
  %3 = stablehlo.constant dense<0.5> : tensor<f32>
  %4 = stablehlo.constant dense<0.6> : tensor<f32>
  %5 = stablehlo.constant dense<1.5> : tensor<f32>
  %6 = stablehlo.constant dense<2.5> : tensor<f32>

  %7 = stablehlo.round_nearest_even %0 : tensor<f32>
  %8 = stablehlo.round_nearest_even %1 : tensor<f32>
  %9 = stablehlo.round_nearest_even %2 : tensor<f32>
  %10 = stablehlo.round_nearest_even %3 : tensor<f32>
  %11 = stablehlo.round_nearest_even %4 : tensor<f32>
  %12 = stablehlo.round_nearest_even %5 : tensor<f32>
  %13 = stablehlo.round_nearest_even %6 : tensor<f32>

  func.return %7, %8, %9, %10, %11, %12, %13 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

////////
// RsqrtOp

// CHECK-LABEL: func @fold_rsqrt
func.func @fold_rsqrt() -> (tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[HALF:%.*]] = stablehlo.constant dense<5.0{{.*}}e-01> : tensor<f32>
  // CHECK-DAG: [[ZERO:%.*]] = stablehlo.constant dense<0.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[DO_NOT_FOLD_RSQRT_ZERO:%.*]] = stablehlo.rsqrt [[ZERO]] : tensor<f32>
  // CHECK:     return [[HALF]], [[DO_NOT_FOLD_RSQRT_ZERO]]

  %0 = stablehlo.constant dense<4.0> : tensor<f32>
  %1 = stablehlo.constant dense<0.0> : tensor<f32>

  %2 = stablehlo.rsqrt %0 : tensor<f32>
  %3 = stablehlo.rsqrt %1 : tensor<f32>

  func.return %2, %3 : tensor<f32>, tensor<f32>
}

// -----

////////
// SignOp

// CHECK-LABEL: func @fold_sign
func.func @fold_sign() -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[INT_NEG_ONE:%.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // CHECK-DAG: [[INT_ZERO:%.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: [[INT_POS_ONE:%.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: [[FLOAT_NEG_ONE:%.*]] = stablehlo.constant dense<-1.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[FLOAT_ZERO:%.*]] = stablehlo.constant dense<0.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[FLOAT_POS_ONE:%.*]] = stablehlo.constant dense<1.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[FLOAT_NAN:%.*]] = stablehlo.constant dense<0x7FC00000> : tensor<f32>
  // CHECK-DAG: [[DO_NOT_FOLD_SIGN_NAN:%.*]] = stablehlo.sign [[FLOAT_NAN]] : tensor<f32>
  // CHECK:     return [[INT_NEG_ONE]], [[INT_ZERO]], [[INT_POS_ONE]], [[FLOAT_NEG_ONE]], [[FLOAT_ZERO]], [[FLOAT_POS_ONE]], [[DO_NOT_FOLD_SIGN_NAN]]

  %0 = stablehlo.constant dense<-7> : tensor<i32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  %2 = stablehlo.constant dense<25> : tensor<i32>
  %3 = stablehlo.constant dense<-2.5> : tensor<f32>
  %4 = stablehlo.constant dense<0.0> : tensor<f32>
  %5 = stablehlo.constant dense<0.1> : tensor<f32>
  %6 = stablehlo.constant dense<0x7FC00000> : tensor<f32> // NaN

  %7 = stablehlo.sign %0 : tensor<i32>
  %8 = stablehlo.sign %1 : tensor<i32>
  %9 = stablehlo.sign %2 : tensor<i32>
  %10 = stablehlo.sign %3 : tensor<f32>
  %11 = stablehlo.sign %4 : tensor<f32>
  %12 = stablehlo.sign %5 : tensor<f32>
  %13 = stablehlo.sign %6 : tensor<f32>

  func.return %7, %8, %9, %10, %11, %12, %13 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

////////
// SineOp

// CHECK-LABEL: func @fold_sine
func.func @fold_sine() -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[ZERO:%.*]] = stablehlo.constant dense<{{0\.0000.*}}> : tensor<f32>
  // CHECK-DAG: [[HALF:%.*]] = stablehlo.constant dense<{{0\.5000.*|5\.000.*[Ee]-01|0.4999.*|4\.999.*[Ee]-01}}> : tensor<f32>
  // CHECK-DAG: [[SQRT_TWO_OVER_TWO:%.*]] = stablehlo.constant dense<{{0\.7071.*|7\.071.*[Ee]-01}}> : tensor<f32>
  // CHECK-DAG: [[SQRT_THREE_OVER_TWO:%.*]] = stablehlo.constant dense<{{0\.8660.*|8\.660.*[Ee]-01}}> : tensor<f32>
  // CHECK-DAG: [[ONE:%.*]] = stablehlo.constant dense<{{1\.0000.*|0\.9999.*|9\.999.*[Ee]-01}}> : tensor<f32>
  // CHECK:     return [[ZERO]], [[HALF]], [[SQRT_TWO_OVER_TWO]], [[SQRT_THREE_OVER_TWO]], [[ONE]]

  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.constant dense<0.5235987755982989> : tensor<f32> // pi/6
  %2 = stablehlo.constant dense<0.7853981633974483> : tensor<f32> // pi/4
  %3 = stablehlo.constant dense<1.0471975511965977> : tensor<f32> // pi/3
  %4 = stablehlo.constant dense<1.5707963267948966> : tensor<f32> // pi/2

  %5 = stablehlo.sine %0 : tensor<f32>
  %6 = stablehlo.sine %1 : tensor<f32>
  %7 = stablehlo.sine %2 : tensor<f32>
  %8 = stablehlo.sine %3 : tensor<f32>
  %9 = stablehlo.sine %4 : tensor<f32>

  func.return %5, %6, %7, %8, %9 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

////////
// SqrtOp

// CHECK-LABEL: func @fold_sqrt
func.func @fold_sqrt() -> (tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[TWO:%.*]] = stablehlo.constant dense<2.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[NEG_ONE:%.*]] = stablehlo.constant dense<-1.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[DO_NOT_FOLD_FLOAT_SQRT_NEG_ONE:%.*]] = stablehlo.sqrt [[NEG_ONE]] : tensor<f32>
  // CHECK:     return [[TWO]], [[DO_NOT_FOLD_FLOAT_SQRT_NEG_ONE]]

  %0 = stablehlo.constant dense<4.0> : tensor<f32>
  %1 = stablehlo.constant dense<-1.0> : tensor<f32>

  %2 = stablehlo.sqrt %0 : tensor<f32>
  %3 = stablehlo.sqrt %1 : tensor<f32>

  func.return %2, %3 : tensor<f32>, tensor<f32>
}

// -----

////////
// TanOp

// CHECK-LABEL: func @fold_tan
func.func @fold_tan() -> (tensor<f32>) {
  // CHECK: [[ONE:%.*]] = stablehlo.constant dense<{{1\.0.*|0\.999.*}}> : tensor<f32>
  // CHECK: return [[ONE]]
  %pi_over_4 = stablehlo.constant dense<0.7853981633974483> : tensor<f32>
  %result = stablehlo.tan %pi_over_4 : tensor<f32>
  func.return %result : tensor<f32>
}

// -----

////////
// TanhOp

// CHECK-LABEL: func @fold_tanh
func.func @fold_tanh() -> (tensor<f32>, tensor<f32>, tensor<f32>) {
  // CHECK-DAG: [[NEG_SQRT_ONE_FIFTH:%.*]] = stablehlo.constant dense<-0.44721{{.*}}> : tensor<f32>
  // CHECK-DAG: [[ZERO:%.*]] = stablehlo.constant dense<0.0{{.*}}> : tensor<f32>
  // CHECK-DAG: [[SQRT_ONE_FIFTH:%.*]] = stablehlo.constant dense<0.44721{{.*}}> : tensor<f32>
  // CHECK:     return [[NEG_SQRT_ONE_FIFTH]], [[ZERO]], [[SQRT_ONE_FIFTH]]

  %neg_log_phi = stablehlo.constant dense<-0.4812118250596034> : tensor<f32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %log_phi = stablehlo.constant dense<0.4812118250596034> : tensor<f32>

  %tanh_neg_log_phi = stablehlo.tanh %neg_log_phi : tensor<f32>
  %tanh_zero = stablehlo.tanh %zero : tensor<f32>
  %tanh_log_phi = stablehlo.tanh %log_phi : tensor<f32>

  func.return %tanh_neg_log_phi, %tanh_zero, %tanh_log_phi : tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

////////
// SetDimensionSizeOp

// CHECK-LABEL: func.func @fold_set_dimension_size
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<10xf32>)
func.func @fold_set_dimension_size(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK-NOT: stablehlo.set_dimension_size
  // CHECK: return [[ARG0]]
  %c = stablehlo.constant dense<10> : tensor<i32>
  %0 = stablehlo.set_dimension_size %arg0, %c, dim = 0 : (tensor<10xf32>, tensor<i32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

// Don't fold when set_dimension_size result is bounded.
// CHECK-LABEL: func.func @no_fold_set_dimension_size
func.func @no_fold_set_dimension_size(%arg0: tensor<10xf32>) -> tensor<?xf32, #stablehlo.bounds<10>> {
  %c = stablehlo.constant dense<10> : tensor<i32>
  // CHECK: [[RESULT0:%.+]] = stablehlo.set_dimension_size
  // CHECK-NEXT: return [[RESULT0]]
  %0 = stablehlo.set_dimension_size %arg0, %c, dim = 0 : (tensor<10xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<10>>
  return %0 : tensor<?xf32, #stablehlo.bounds<10>>
}

// -----

// Don't fold when washing away a bounded dimension, not safe to replace with
// operand when types mismatch.
// CHECK-LABEL: func.func @no_fold_set_dimension_size_bounded_input
func.func @no_fold_set_dimension_size_bounded_input(%arg0: tensor<?x4xf32, #stablehlo.bounds<8, ?>>) -> tensor<8x4xf32> {
  %c = stablehlo.constant dense<8> : tensor<i32>
  // CHECK: [[RESULT0:%.+]] = stablehlo.set_dimension_size
  // CHECK-NEXT: return [[RESULT0]]
  %0 = stablehlo.set_dimension_size %arg0, %c, dim = 0 : (tensor<?x4xf32, #stablehlo.bounds<8, ?>>, tensor<i32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// -----

////////
// TransposeOp

// CHECK-LABEL: func @eval_transpose
func.func @eval_transpose() -> (tensor<2x3x2xi32>, tensor<2x4x3xi32>, tensor<4x3x2xi32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[1, 7], [3, 9], [5, 11]],
  // CHECK-SAME:   {{\[}}[2, 8], [4, 10], [6, 12]]]> : tensor<2x3x2xi32>
  //
  // CHECK: [[RESULT1:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[1, 3, 5], [7, 9, 11], [13, 15, 17], [19, 21, 23]],
  // CHECK-SAME:   {{\[}}[2, 4, 6], [8, 10, 12], [14, 16, 18], [20, 22, 24]]]> : tensor<2x4x3xi32>
  //
  // CHECK: [[RESULT2:%.*]] = stablehlo.constant dense<
  // CHECK-SAME: {{\[\[}}[1, 2],  [3, 4],  [5, 6]]
  // CHECK-SAME:   {{\[}}[7, 8],  [9, 10], [11, 12]],
  // CHECK-SAME:   {{\[}}[13, 14], [15, 16], [17, 18]],
  // CHECK-SAME:   {{\[}}[19, 20], [21, 22], [23, 24]]]> : tensor<4x3x2xi32>
  //
  // CHECK: return [[RESULT0]], [[RESULT1]], [[RESULT2]]
  %0 = stablehlo.constant dense<[[[1,2], [3,4], [5,6]],
                                 [[7,8], [9,10], [11,12]]]> : tensor<2x3x2xi32>
  %1 = stablehlo.constant dense<[[[1, 2],  [3, 4],  [5, 6]],
                                 [[7, 8],  [9, 10], [11,12]],
                                 [[13,14], [15,16], [17,18]],
                                 [[19,20], [21,22], [23,24]]]> : tensor<4x3x2xi32>
  %2 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  %3 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<4x3x2xi32>) -> tensor<2x4x3xi32>
  %4 = stablehlo.transpose %3, dims = [1, 2, 0] : (tensor<2x4x3xi32>) -> tensor<4x3x2xi32>
  func.return %2, %3, %4 : tensor<2x3x2xi32>, tensor<2x4x3xi32>, tensor<4x3x2xi32>
}

// -----

// CHECK-LABEL: func @eval_transpose_zerodim
func.func @eval_transpose_zerodim() -> (tensor<10x3x0xf32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<> : tensor<10x3x0xf32>
  // CHECK: return [[RESULT0]]
  %0 = stablehlo.constant dense<> : tensor<3x0x10xf32>
  %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<3x0x10xf32>) -> tensor<10x3x0xf32>
  func.return %1 : tensor<10x3x0xf32>
}

// -----

// CHECK-LABEL: func @eval_transpose_zerorank
func.func @eval_transpose_zerorank() -> tensor<i32> {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK: return [[RESULT0]]
  %0 = stablehlo.constant dense<1> : tensor<i32>
  %1 = stablehlo.transpose %0, dims = [] : (tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: func @eval_transpose_splat
func.func @eval_transpose_splat() -> (tensor<10x3x1xi32>) {
  // CHECK: [[RESULT0:%.*]] = stablehlo.constant dense<1> : tensor<10x3x1xi32>
  // CHECK: return [[RESULT0]]
  %0 = stablehlo.constant dense<1> : tensor<3x1x10xi32>
  %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<3x1x10xi32>) -> tensor<10x3x1xi32>
  func.return %1 : tensor<10x3x1xi32>
}

// -----

////////
// WhileOp

// CHECK-LABEL: dce_while_false_condition
func.func @dce_while_false_condition() -> tensor<i64> {
  %0 = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NOT: stablehlo.while
  %1 = stablehlo.while(%iterArg = %0) : tensor<i64>
    cond {
    %2 = stablehlo.compare LT, %0, %0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  } do {
    %2 = stablehlo.custom_call @something(%iterArg) {has_side_effect = false} : (tensor<i64>) -> tensor<i64>
    stablehlo.return %2 : tensor<i64>
  }
  return %1 : tensor<i64>
}
