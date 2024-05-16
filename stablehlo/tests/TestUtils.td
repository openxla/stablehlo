/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

include "mlir/Pass/PassBase.td"

def HloTestInferPass : Pass<"hlo-test-infer", "func::FuncOp"> {
  let summary = "Uses test ops to invoke InferShapedTypeOpInterface methods.";
  let dependentDialects = ["shape::ShapeDialect"];
}

def HloTestSpeculatabilityPass : Pass<"hlo-test-speculatability", "func::FuncOp"> {
  let summary = "Uses test ops to invoke ConditionallySpeculatable::getSpeculatability.";
  let description = [{
    The hlo-test-speculatability pass looks for ops like "hlo_test_speculatability.is_speculatable"(%arg)
    and "hlo_test_speculatability.is_not_speculatable"(%arg) and checks that
    their single operand's defining op has the expected speculatability. If it
    does, the op is deleted, otherwise the op is untouched.
  }];
  let dependentDialects = [];
}
