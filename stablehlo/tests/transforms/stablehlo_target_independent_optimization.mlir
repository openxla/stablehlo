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

func.func @concatenate_fold_splat_flatten_integ(%arg0: tensor<8xf32>) -> tensor<64xf32> {
  // CHECK-DAG: [[CST0:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<8xf32>
  // CHECK-DAG: [[CST1:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<8xf32>
  // CHECK-DAG: [[CST2:%.+]] = stablehlo.constant dense<2.000000e+00> : tensor<8xf32>
  // CHECK-DAG: [[CST3:%.+]] = stablehlo.constant dense<3.000000e+00> : tensor<8xf32>
  // CHECK: stablehlo.concatenate [[CST0]], [[CST1]], [[CST2]], [[CST3]], %arg0, %arg0, %arg0, %arg0,
  %cst0 = stablehlo.constant dense<0.0> : tensor<f32>
  %cst1 = stablehlo.constant dense<1.0> : tensor<f32>
  %cst2 = stablehlo.constant dense<2.0> : tensor<f32>
  %cst3 = stablehlo.constant dense<3.0> : tensor<f32>
  %0 = stablehlo.reshape %cst0 : (tensor<f32>) -> tensor<1xf32>
  %1 = stablehlo.reshape %cst1 : (tensor<f32>) -> tensor<1xf32>
  %2 = stablehlo.reshape %cst2 : (tensor<f32>) -> tensor<1xf32>
  %3 = stablehlo.reshape %cst3 : (tensor<f32>) -> tensor<1xf32>
  %4 = stablehlo.concatenate %0, %0, %0, %0, %0, %0, %0, %0, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<8xf32>
  %5 = stablehlo.concatenate %1, %1, %1, %1, %1, %1, %1, %1, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<8xf32>
  %6 = stablehlo.concatenate %2, %2, %2, %2, %2, %2, %2, %2, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<8xf32>
  %7 = stablehlo.concatenate %3, %3, %3, %3, %3, %3, %3, %3, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<8xf32>
  %8 = stablehlo.concatenate %4, %5, %6, %7, dim = 0 : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<32xf32>
  %9 = stablehlo.concatenate %arg0, %arg0, %arg0, %arg0, dim = 0 : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<32xf32>
  %10 = stablehlo.concatenate %8, %9, dim = 0 : (tensor<32xf32>, tensor<32xf32>) -> tensor<64xf32>
  return %10 : tensor<64xf32>
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
