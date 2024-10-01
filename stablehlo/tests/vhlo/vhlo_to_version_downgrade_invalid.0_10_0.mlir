// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.10.0' --verify-diagnostics --split-input-file %s

// expected-error @-3 {{failed to convert VHLO to v0.10.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_fp8_E4M3B11FNUZ(%arg0: tensor<f8E4M3B11FNUZ>) -> tensor<f8E4M3B11FNUZ> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<f8E4M3B11FNUZ>
  func.return %0 : tensor<f8E4M3B11FNUZ>
}
