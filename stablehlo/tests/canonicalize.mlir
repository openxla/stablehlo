// RUN: stablehlo-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' -split-input-file -allow-unregistered-dialect | FileCheck %s

func.func @fold_constant_case(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
 %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
 %1 = "stablehlo.case"(%0) ({
   "stablehlo.return"(%arg0) : (tensor<i32>) -> ()
 }, {
  "stablehlo.return"(%arg1) : (tensor<i32>) -> ()
 }) : (tensor<i32>) -> tensor<i32>
 return %1 : tensor<i32>

// CHECK: return %arg1
}
