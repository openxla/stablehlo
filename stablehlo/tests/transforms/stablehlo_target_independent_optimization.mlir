// RUN: stablehlo-opt --stablehlo-target-independent-optimization --split-input-file %s | FileCheck %s

// Check that simplificaiton and folding are both applied.

// CHECK-LABEL: @add_cst_on_rhs
func.func @add_cst_on_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<1.0> : tensor<f32>
  %0 = stablehlo.add %cst, %cst : tensor<f32>
  // CHECK: stablehlo.add %arg0, %cst : tensor<f32>
  %1 = stablehlo.add %0, %arg0 : tensor<f32>
  return %1 : tensor<f32>
}

// -----

// Check that the WhileOp optimizations from AggressiveFolder and
// AggressiveSimplification don't interfere with one another. Specifically, we
// need to make sure that FoldWhileOpPattern doesn't DCE code with side effects.

// CHECK-LABEL: while_op_with_outfeed_no_dce
func.func @while_op_with_outfeed_no_dce(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK: stablehlo.while
  %0 = stablehlo.while(%iterArg = %arg0) : tensor<i64>
    cond {
    %1 = stablehlo.compare LT, %iterArg, %iterArg, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
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
    %1 = stablehlo.compare LT, %iterArg, %iterArg, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.create_token : !stablehlo.token
    stablehlo.return %iterArg : tensor<i64>
  }
  return %arg0 : tensor<i64>
}

// CHECK-LABEL: dce_while_false_condition
func.func @dce_while_false_condition(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK-NOT: stablehlo.while
  %0 = stablehlo.while(%iterArg = %arg0) : tensor<i64>
    cond {
    %1 = stablehlo.compare LT, %arg0, %arg0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.custom_call @something(%iterArg) {has_side_effect = false} : (tensor<i64>) -> tensor<i64>
    stablehlo.return %1 : tensor<i64>
  }
  return %0 : tensor<i64>
}

// -----

// Check that we properly handle expressions involving NaN terms or variables
// that could potentially be NaN.

// CHECK-LABEL: @fold_constant_nan_to_nan
func.func @fold_constant_nan_to_nan() -> tensor<f32> {
  // CHECK: [[NAN:%.*]] = stablehlo.constant dense<0x7FC00000> : tensor<f32>
  // CHECK: return [[NAN]] : tensor<f32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %one = stablehlo.constant dense<1.0> : tensor<f32>
  %nan = stablehlo.constant dense<0x7FC00000> : tensor<f32>
  %nan_times_zero = stablehlo.multiply %nan, %zero : tensor<f32>
  %result = stablehlo.add %one, %nan_times_zero : tensor<f32>
  return %result : tensor<f32>
}

// TODO: Consider adding an `--assume-non-nan` pass option to override this.
// CHECK-LABEL: @do_not_assume_non_nan
func.func @do_not_assume_non_nan(%arg0: tensor<f32>) -> tensor<f32> {
  // Note: These two checks are out of order on purpose: [[RESULT]] binds to the
  // `return` op first and then looks backward for the corresponding assignment.
  // CHECK-DAG: return [[RESULT:.*]] : tensor<f32>
  // CHECK-DAG: [[RESULT]] = stablehlo.{{(add|multiply).*}} : tensor<f32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %one = stablehlo.constant dense<1.0> : tensor<f32>
  %arg_times_zero = stablehlo.multiply %arg0, %zero : tensor<f32>
  %result = stablehlo.add %one, %arg_times_zero : tensor<f32>
  return %result : tensor<f32>
}
