// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.1.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_i2(%arg0: tensor<i2>) -> tensor<i2> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<i2>
  func.return %0 : tensor<i2>
}

// -----

// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_ui2(%arg0: tensor<ui2>) -> tensor<ui2> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<ui2>
  func.return %0 : tensor<ui2>
}

// -----

func.func @custom_call_dictionary_attr(%arg0: tensor<f32>) -> tensor<f32> {
// expected-error @+1 {{failed to legalize operation 'stablehlo.custom_call' that was explicitly marked illegal}}
%0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    api_version = 4 : i32,
    backend_config={foo = 42 : i32}
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
