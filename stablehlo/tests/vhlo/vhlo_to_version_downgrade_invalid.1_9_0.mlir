// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.9.0' --verify-diagnostics --split-input-file %s


func.func @cbrt_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.cbrt"(%arg0) {
    // CHECK: vhlo.cbrt_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.9.0}}
module {
func.func @cbrt_invalid(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.cbrt_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.cbrt"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

// -----

func.func @cosine_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.cosine"(%arg0) {
    // CHECK: vhlo.cosine_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.9.0}}
module {
func.func @cosine_invalid(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.cosine_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.cosine"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

// -----

func.func @exponential_minus_one_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.exponential_minus_one"(%arg0) {
    // CHECK: vhlo.exponential_minus_one_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.9.0}}
module {
func.func @exponential_minus_one_invalid(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.exponential_minus_one_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.exponential_minus_one"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

// -----

func.func @log_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.log"(%arg0) {
    // CHECK: vhlo.log_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.9.0}}
module {
func.func @log_invalid(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.log_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.log"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

// -----

func.func @log_plus_one_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.log_plus_one"(%arg0) {
    // CHECK: vhlo.log_plus_one_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.9.0}}
module {
func.func @log_plus_one_invalid(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.log_plus_one_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.log_plus_one"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

// -----

func.func @logistic_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.logistic"(%arg0) {
    // CHECK: vhlo.logistic_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.9.0}}
module {
func.func @logistic_invalid(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.logistic_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.logistic"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

// -----

func.func @rsqrt_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.rsqrt"(%arg0) {
    // CHECK: vhlo.rsqrt_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.9.0}}
module {
func.func @rsqrt_invalid(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.rsqrt_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.rsqrt"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

// -----

func.func @sine_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.sine"(%arg0) {
    // CHECK: vhlo.sine_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.9.0}}
module {
func.func @sine_invalid(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.sine_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.sine"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

// -----

func.func @sqrt_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.sqrt"(%arg0) {
    // CHECK: vhlo.sqrt_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.9.0}}
module {
func.func @sqrt_invalid(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.sqrt_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.sqrt"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

// -----

func.func @tan_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.tan"(%arg0) {
    // CHECK: vhlo.tan_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.9.0}}
module {
func.func @tan_invalid(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.tan_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.tan"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

// -----

func.func @tanh_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.tanh"(%arg0) {
    // CHECK: vhlo.tanh_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.9.0}}
module {
func.func @tanh_invalid(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.tanh_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.tanh"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

