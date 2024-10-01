// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.1.0' --verify-diagnostics --split-input-file %s

// expected-error @-3 {{failed to convert VHLO to v1.1.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_i2(%arg0: tensor<i2>) -> tensor<i2> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<i2>
  func.return %0 : tensor<i2>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v1.1.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_ui2(%arg0: tensor<ui2>) -> tensor<ui2> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<ui2>
  func.return %0 : tensor<ui2>
}
