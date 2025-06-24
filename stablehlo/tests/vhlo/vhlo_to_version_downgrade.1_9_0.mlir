// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.9.0' %s | FileCheck %s

// ExpOp was changed in  v1.10.0 to have
// result_accuracy attribute. Ensure that serializing for 1.9.0 is valid and targets the
// v1.9.0 opset.
//
// This will catch issues in op `isLegal` checks:
//   op.minVersion() <= target <= op.maxVersion()

// CHECK-LABEL: vhlo.func_v1 @cbrt_op
func.func public @cbrt_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.cbrt_v1
  %0 = "stablehlo.cbrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @cbrt_default
func.func @cbrt_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.cbrt"(%arg0) {
    // CHECK: vhlo.cbrt_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @cosine_op
func.func public @cosine_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.cosine_v1
  %0 = "stablehlo.cosine"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @cosine_op_unregistered_attrs
func.func public @cosine_op_unregistered_attrs(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.cosine_v1
  // CHECK-SAME: some.unregistered_attr
  %0 = "stablehlo.cosine"(%arg0) { some.unregistered_attr = 1 : i32 } : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @cosine_default
func.func @cosine_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.cosine"(%arg0) {
    // CHECK: vhlo.cosine_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: vhlo.func_v1 @exponential_minus_one_op
func.func public @exponential_minus_one_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.exponential_minus_one_v1
  %0 = "stablehlo.exponential_minus_one"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @exponential_minus_one_default
func.func @exponential_minus_one_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.exponential_minus_one"(%arg0) {
    // CHECK: vhlo.exponential_minus_one_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @log_op
func.func public @log_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.log_v1
  %0 = "stablehlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @log_default
func.func @log_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.log"(%arg0) {
    // CHECK: vhlo.log_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @log_plus_one_op
func.func public @log_plus_one_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.log_plus_one_v1
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @log_plus_one_default
func.func @log_plus_one_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.log_plus_one"(%arg0) {
    // CHECK: vhlo.log_plus_one_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @logistic_op
func.func public @logistic_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.logistic_v1
  %0 = "stablehlo.logistic"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @logistic_default
func.func @logistic_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.logistic"(%arg0) {
    // CHECK: vhlo.logistic_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @rsqrt_op
func.func public @rsqrt_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.rsqrt_v1
  %0 = "stablehlo.rsqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @rsqrt_default
func.func @rsqrt_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.rsqrt"(%arg0) {
    // CHECK: vhlo.rsqrt_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @sine_op
func.func public @sine_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.sine_v1
  %0 = "stablehlo.sine"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @sine_default
func.func @sine_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.sine"(%arg0) {
    // CHECK: vhlo.sine_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @sqrt_op
func.func public @sqrt_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.sqrt_v1
  %0 = "stablehlo.sqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @sqrt_default
func.func @sqrt_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.sqrt"(%arg0) {
    // CHECK: vhlo.sqrt_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @tan_op
func.func public @tan_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.tan_v1
  %0 = "stablehlo.tan"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @tan_default
func.func @tan_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.tan"(%arg0) {
    // CHECK: vhlo.tan_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @tanh_op
func.func public @tanh_op(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: vhlo.tanh_v1
  %0 = "stablehlo.tanh"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: vhlo.func_v1 @tanh_default
func.func @tanh_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.tanh"(%arg0) {
    // CHECK: vhlo.tanh_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

