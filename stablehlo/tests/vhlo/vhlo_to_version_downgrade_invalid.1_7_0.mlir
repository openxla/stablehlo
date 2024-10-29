// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.7.0' --verify-diagnostics --split-input-file %s

// expected-error @-3 {{failed to convert VHLO to v1.7.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_f4E2M1FN(%arg0: tensor<f4E2M1FN>, %arg1: tensor<f4E2M1FN>) -> tensor<f4E2M1FN> {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f4E2M1FN>, tensor<f4E2M1FN>) -> tensor<f4E2M1FN>
  func.return %0 : tensor<f4E2M1FN>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v1.7.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_f6E2M3FN(%arg0: tensor<f6E2M3FN>, %arg1: tensor<f6E2M3FN>) -> tensor<f6E2M3FN> {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f6E2M3FN>, tensor<f6E2M3FN>) -> tensor<f6E2M3FN>
  func.return %0 : tensor<f6E2M3FN>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v1.7.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_f6E3M2FN(%arg0: tensor<f6E3M2FN>, %arg1: tensor<f6E3M2FN>) -> tensor<f6E3M2FN> {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f6E3M2FN>, tensor<f6E3M2FN>) -> tensor<f6E3M2FN>
  func.return %0 : tensor<f6E3M2FN>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v1.7.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_f8E8M0FNU(%arg0: tensor<f8E8M0FNU>, %arg1: tensor<f8E8M0FNU>) -> tensor<f8E8M0FNU> {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E8M0FNU>, tensor<f8E8M0FNU>) -> tensor<f8E8M0FNU>
  func.return %0 : tensor<f8E8M0FNU>
}
