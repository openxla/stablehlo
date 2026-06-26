// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.17.0' %s | FileCheck %s

// CustomCallOp was changed in v1.18.0 to have result_tilings attribute. Ensure that serializing for 1.17.0 is valid and targets the v1.17.0 opset when result_tilings is empty or omitted.

// CHECK-LABEL: vhlo.func_v1 @custom_call_default
func.func @custom_call_default(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.custom_call_v1
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo"
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @custom_call_explicit_empty
func.func @custom_call_explicit_empty(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.custom_call_v1
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    result_tilings = []
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
