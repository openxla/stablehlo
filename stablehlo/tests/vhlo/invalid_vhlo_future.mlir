// RUN: not stablehlo-translate --deserialize %s.bc --verify-diagnostics 2>&1 | FileCheck %s
//
// Note: This file is not valid to parse since VHLO doesn't support unknown
// operations, but is kept around to help visualize the bytecode file in test.
// The bytecode file should not break, as it is a portable artifact with full
// backward compatibility.

// CHECK: error: unregistered operation 'vhlo.constant_v99' found in dialect ('vhlo') that does not allow unknown operations
// CHECK: note: in bytecode version 6 produced by: StableHLO_v2.0.0
// CHECK: error: failed to deserialize portable artifact using StableHLO_v{{.*}}
vhlo.func_v1 @main() -> (!vhlo.tensor_v1<!vhlo.f32_v1>) {
  %0 = "vhlo.constant_v99"() <{value = #vhlo.tensor_v1<dense<1.000000e+00> : tensor<f32>>}> : () -> !vhlo.tensor_v1<!vhlo.f32_v1>
  "vhlo.return_v1"(%0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
} {arg_attrs = #vhlo.array_v1<[]>, res_attrs = #vhlo.array_v1<[]>, sym_visibility = #vhlo.string_v1<"public">}
