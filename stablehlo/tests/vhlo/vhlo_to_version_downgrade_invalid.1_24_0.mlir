// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.24.0' --verify-diagnostics --split-input-file %s

// expected-error @+2 {{failed to convert VHLO to v1.24.0}}
// expected-error @+2 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
module {
  func.func @type_f8E5M3FNU(%arg0: tensor<f8E5M3FNU>, %arg1: tensor<f8E5M3FNU>) -> tensor<f8E5M3FNU> {
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E5M3FNU>, tensor<f8E5M3FNU>) -> tensor<f8E5M3FNU>
    func.return %0 : tensor<f8E5M3FNU>
  }
}
