// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.9.0' --verify-diagnostics --split-input-file %s

// expected-error @-3 {{failed to convert VHLO to v0.9.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_fp8_E5M2FNUZ(%arg0: tensor<f8E5M2FNUZ>) -> tensor<f8E5M2FNUZ> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<f8E5M2FNUZ>
  func.return %0 : tensor<f8E5M2FNUZ>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v0.9.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_fp8_E4M3FNUZ(%arg0: tensor<f8E4M3FNUZ>) -> tensor<f8E4M3FNUZ> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<f8E4M3FNUZ>
  func.return %0 : tensor<f8E4M3FNUZ>
}
