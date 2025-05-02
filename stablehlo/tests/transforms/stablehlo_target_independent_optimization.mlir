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
