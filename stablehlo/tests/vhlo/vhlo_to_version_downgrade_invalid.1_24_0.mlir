// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.24.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  func.func @complex_operand_f16(%arg0: tensor<4xf16>) -> tensor<4xcomplex<f16>> {
    // expected-error @+1 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
    %0 = "stablehlo.convert"(%arg0) : (tensor<4xf16>) -> tensor<4xcomplex<f16>>
    // expected-error @+1 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
    func.return %0 : tensor<4xcomplex<f16>>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  func.func @complex_tensor_element_f16(%arg0: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
    // expected-error @+1 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
    %0 = "stablehlo.negate"(%arg0) : (tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
    // expected-error @+1 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
    func.return %0 : tensor<4xcomplex<f16>>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.bf16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  func.func @complex_bf16(%arg0: tensor<2xcomplex<bf16>>, %arg1: tensor<2xcomplex<bf16>>) -> tensor<2xcomplex<bf16>> {
    // expected-error @+1 {{failed to legalize type '!vhlo.complex_v1<!vhlo.bf16_v1>' for target version 1.24.0}}
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2xcomplex<bf16>>, tensor<2xcomplex<bf16>>) -> tensor<2xcomplex<bf16>>
    // expected-error @+1 {{failed to legalize type '!vhlo.complex_v1<!vhlo.bf16_v1>' for target version 1.24.0}}
    func.return %0 : tensor<2xcomplex<bf16>>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  func.func @complex_tuple_element_f16(%arg0: tensor<2xf16>) -> tuple<tensor<2xcomplex<f16>>> {
    // expected-error @+1 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
    %0 = "stablehlo.custom_call"(%arg0) { call_target_name = "foo" } : (tensor<2xf16>) -> tuple<tensor<2xcomplex<f16>>>
    // expected-error @+1 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
    func.return %0 : tuple<tensor<2xcomplex<f16>>>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  func.func @complex_buffer_element_f16(%arg0: memref<4xcomplex<f16>>) -> memref<4xcomplex<f16>> {
    // expected-error @+1 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
    func.return %arg0 : memref<4xcomplex<f16>>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  func.func private @complex_func_signature_f16() -> tensor<2xcomplex<f16>>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  func.func @complex_type_attr_f16() attributes { stablehlo.attr = complex<f16> } {
    return
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  func.func private @complex_future_element_f16() -> !stablehlo.future<tensor<2xcomplex<f16>>>
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  func.func @complex_tensor_attr_f16() -> tensor<2xcomplex<f16>> {
    // expected-error @+1 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
    %0 = stablehlo.constant dense<[(1.0, 2.0), (3.0, 4.0)]> : tensor<2xcomplex<f16>>
    // expected-error @+1 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
    func.return %0 : tensor<2xcomplex<f16>>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  "vhlo.func_v1"() <{
    arg_attrs = #vhlo.array_v1<[]>,
    function_type = #vhlo.type_v1<!vhlo.func_v1<(()) -> !vhlo.unranked_tensor_v1<!vhlo.complex_v1<!vhlo.f16_v1>>>>,
    res_attrs = #vhlo.array_v1<[]>,
    sym_name = #vhlo.string_v1<"complex_unranked_element_f16">,
    sym_visibility = #vhlo.string_v1<"private">
  }> ({
    ^bb0:
      "vhlo.return_v1"() : () -> ()
  }) : () -> ()
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  "vhlo.func_v1"() <{
    arg_attrs = #vhlo.array_v1<[]>,
    function_type = #vhlo.type_v1<!vhlo.func_v1<(()) -> !vhlo.tensor_v1<!vhlo.quant_v1<!vhlo.i8_v1:!vhlo.complex_v1<!vhlo.f16_v1>, 1.000000e+00:0, 0:0, 0>>>>,
    res_attrs = #vhlo.array_v1<[]>,
    sym_name = #vhlo.string_v1<"complex_quant_expressed_f16">,
    sym_visibility = #vhlo.string_v1<"private">
  }> ({
    ^bb0:
      "vhlo.return_v1"() : () -> ()
  }) : () -> ()
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  "vhlo.func_v1"() <{
    arg_attrs = #vhlo.array_v1<[]>,
    function_type = #vhlo.type_v1<!vhlo.func_v1<(()) -> !vhlo.tensor_v1<!vhlo.quant_v1<!vhlo.complex_v1<!vhlo.f16_v1>:!vhlo.f32_v1, 1.000000e+00:0, 0:0, 0>>>>,
    res_attrs = #vhlo.array_v1<[]>,
    sym_name = #vhlo.string_v1<"complex_quant_storage_f16">,
    sym_visibility = #vhlo.string_v1<"private">
  }> ({
    ^bb0:
      "vhlo.return_v1"() : () -> ()
  }) : () -> ()
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  "vhlo.func_v1"() <{
    arg_attrs = #vhlo.array_v1<[]>,
    function_type = #vhlo.type_v1<!vhlo.func_v1<(()) -> !vhlo.tensor_v1<!vhlo.quant_per_axis_v1<!vhlo.i8_v1:!vhlo.complex_v1<!vhlo.f16_v1>, 0, [1.000000e+00], [0], 0:0, 0>>>>,
    res_attrs = #vhlo.array_v1<[]>,
    sym_name = #vhlo.string_v1<"complex_quant_per_axis_expressed_f16">,
    sym_visibility = #vhlo.string_v1<"private">
  }> ({
    ^bb0:
      "vhlo.return_v1"() : () -> ()
  }) : () -> ()
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
  "vhlo.func_v1"() <{
    arg_attrs = #vhlo.array_v1<[]>,
    function_type = #vhlo.type_v1<!vhlo.func_v1<(()) -> !vhlo.tensor_v1<!vhlo.quant_per_axis_v1<!vhlo.complex_v1<!vhlo.f16_v1>:!vhlo.f32_v1, 0, [1.000000e+00], [0], 0:0, 0>>>>,
    res_attrs = #vhlo.array_v1<[]>,
    sym_name = #vhlo.string_v1<"complex_quant_per_axis_storage_f16">,
    sym_visibility = #vhlo.string_v1<"private">
  }> ({
    ^bb0:
      "vhlo.return_v1"() : () -> ()
  }) : () -> ()
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f8E4M3FN_v1>' for target version 1.24.0}}
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

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.i32_v1>' for target version 1.24.0}}
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

// -----

// expected-error @+1 {{failed to convert VHLO to v1.24.0}}
module {
  func.func private @my_op(tensor<1xf32>) -> tensor<1xf32>
  func.func @complex_region_block_arg_f16(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    // expected-error @+2 {{failed to legalize type '!vhlo.complex_v1<!vhlo.f16_v1>' for target version 1.24.0}}
    // expected-error @+1 {{failed to legalize operation 'vhlo.composite_v2' that was explicitly marked illegal}}
    %0 = "stablehlo.composite"(%arg0) ({
      ^bb0(%arg1: tensor<1xf32>, %arg2: tensor<1xcomplex<f16>>):
        "stablehlo.return"(%arg1) : (tensor<1xf32>) -> ()
    }) {
      name = "my.op",
      decomposition = @my_op,
      version = 1 : i32
    } : (tensor<1xf32>) -> tensor<1xf32>
    func.return %0 : tensor<1xf32>
  }
}
