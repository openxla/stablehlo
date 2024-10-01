// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.6.0' --verify-diagnostics --split-input-file %s

// expected-error @-3 {{failed to convert VHLO to v1.6.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_f8E4M3(%arg0: tensor<f8E4M3>) -> tensor<f8E4M3> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<f8E4M3>
  func.return %0 : tensor<f8E4M3>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v1.6.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_f8E3M4(%arg0: tensor<f8E3M4>) -> tensor<f8E3M4> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<f8E3M4>
  func.return %0 : tensor<f8E3M4>
}
