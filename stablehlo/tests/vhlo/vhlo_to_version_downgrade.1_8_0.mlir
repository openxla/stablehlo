// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.8.0' %s | FileCheck %s

// ExpOp was changed in  v1.9.0 to have
// result_accuracy attribute. Ensure that serializing for 1.8.0 is valid and targets the
// v1.8.0 opset.
//
// This will catch issues in op `isLegal` checks:
//   op.minVersion() <= target <= op.maxVersion()

// CHECK-LABEL: vhlo.func_v1 @exp_op
func.func public @exp_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.exponential_v1
  %0 = "stablehlo.exponential"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: vhlo.func_v1 @exp_op_default
func.func @exp_op_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.exponential"(%arg0) {
    // CHECK: vhlo.exponential_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: vhlo.func_v1 @exp_op_default_unregistered_attrs
func.func @exp_op_default_unregistered_attrs(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.exponential"(%arg0) {
    // CHECK: vhlo.exponential_v1
    // CHECK-SAME: some.unregistered_attr
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>,
    some.unregistered_attr = 1 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
