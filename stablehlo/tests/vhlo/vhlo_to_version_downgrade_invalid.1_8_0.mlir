// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.8.0' --verify-diagnostics --split-input-file %s


func.func @attr_result_accuracy_default(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.exponential"(%arg0) {
    // CHECK: vhlo.exponential_v1
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.8.0}}
module {
func.func @attr_result_accuracy_highest(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.exponential_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.exponential"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
}

