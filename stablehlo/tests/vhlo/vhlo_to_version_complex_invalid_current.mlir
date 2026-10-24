// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.25.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.25.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f8E4M3FN_v1>' for target version 1.25.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  "vhlo.func_v1"() <{
    arg_attrs = #vhlo.array_v1<[]>,
    function_type = #vhlo.type_v1<!vhlo.func_v1<(!vhlo.tensor_v1<!vhlo.complex_v1<!vhlo.f8E4M3FN_v1>>) -> ()>>,
    res_attrs = #vhlo.array_v1<[]>,
    sym_name = #vhlo.string_v1<"complex_operand_f8E4M3FN">,
    sym_visibility = #vhlo.string_v1<"private">
  }> ({
    ^bb0:
      "vhlo.return_v1"() : () -> ()
  }) : () -> ()
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.25.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.i32_v1>' for target version 1.25.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  "vhlo.func_v1"() <{
    arg_attrs = #vhlo.array_v1<[]>,
    function_type = #vhlo.type_v1<!vhlo.func_v1<(()) -> !vhlo.tensor_v1<!vhlo.complex_v1<!vhlo.i32_v1>>>>,
    res_attrs = #vhlo.array_v1<[]>,
    sym_name = #vhlo.string_v1<"complex_result_i32">,
    sym_visibility = #vhlo.string_v1<"private">
  }> ({
    ^bb0:
      "vhlo.return_v1"() : () -> ()
  }) : () -> ()
}
