// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --split-input-file -verify-diagnostics %s

func.func @string_with_type(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'stablehlo.custom_call' that was explicitly marked illegal}}
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo" : tensor<3xf32>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Need to support sparse. Move once forked to VHLO.
// GH Issue: https://github.com/openxla/stablehlo/issues/907

// expected-error @+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @type_sparsity(%arg0: tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<16xf32> {
  // CHECK_DISABLED: "vhlo.abs"(%arg0) : (tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> !vhlo.tensor<tensor<16xf32>>
  %0 = "stablehlo.abs"(%arg0) : (tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-DISABLED-LABEL: "type_sparsity"
