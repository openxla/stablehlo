// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.2.0' --verify-diagnostics --split-input-file %s

// expected-error @-3 {{failed to convert VHLO to v1.2.0}}
func.func @custom_call_dictionary_attr(%arg0: tensor<f32>) -> tensor<f32> {
// expected-error @+1 {{failed to legalize operation 'vhlo.custom_call_v1' that was explicitly marked illegal}}
%0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    api_version = 4 : i32,
    backend_config={foo = 42 : i32}
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v1.2.0}}
func.func @custom_call_dictionary_attr(%arg0: tensor<f32>) -> tensor<f32> {
// expected-error @+1 {{failed to legalize operation 'vhlo.custom_call_v1' that was explicitly marked illegal}}
%0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    api_version = 4 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
