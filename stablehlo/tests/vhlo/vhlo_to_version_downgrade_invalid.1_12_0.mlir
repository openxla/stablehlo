// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.12.0' --verify-diagnostics %s

// expected-error @-3 {{failed to convert VHLO to v1.12.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_buffer(%arg0: memref<2xf32>) -> memref<2xf32> {
  func.return %arg0 : memref<2xf32>
}
