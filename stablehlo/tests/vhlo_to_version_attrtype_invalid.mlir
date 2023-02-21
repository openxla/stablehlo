// RUN: stablehlo-opt --vhlo-to-version='target=0.3.0' --verify-diagnostics --split-input-file %s

// This file tests that legality checks recurse into attributes and types.
// This is needed in case a new type is added to StableHLO, we need to ensure
// that these are prevented from targeting previous versions where the nested type
// or attribute is not supported.

// NOTE: These tests can all be converted to use !vhlo.FP8* once support has been added.
// In the meantime, using an unconverted i16.


// expected-error @+1 {{failed to legalize operation 'vhlo.func' that was explicitly marked illegal}}
vhlo.func @illegal_type_tensor_element(%arg0: !vhlo.tensor<4x16x!vhlo.f8E5M2>) -> () {
  %0 = "vhlo.add"(%arg0, %arg0) : (!vhlo.tensor<4x16x!vhlo.f8E5M2>, !vhlo.tensor<4x16x!vhlo.f8E5M2>) -> !vhlo.tensor<4x16x!vhlo.f8E5M2>
  "vhlo.return"() : () -> ()
}

// -----

// expected-error @+1 {{failed to legalize operation 'vhlo.func' that was explicitly marked illegal}}
vhlo.func @illegal_type_unranked_tensor_element(%arg0: !vhlo.unranked_tensor<!vhlo.f8E4M3FN>) -> () {
  %0 = "vhlo.add"(%arg0, %arg0) : (!vhlo.unranked_tensor<!vhlo.f8E4M3FN>, !vhlo.unranked_tensor<!vhlo.f8E4M3FN>) -> !vhlo.unranked_tensor<!vhlo.f8E4M3FN>
  "vhlo.return"() : () -> ()
}

// -----

// expected-error @+1 {{failed to legalize operation 'vhlo.func' that was explicitly marked illegal}}
vhlo.func @illegal_type_complex_element(%arg0: !vhlo.tensor<4x16x!vhlo.complex<!vhlo.f8E5M2>>) -> () {
  %0 = "vhlo.add"(%arg0, %arg0) : (!vhlo.tensor<4x16x!vhlo.complex<!vhlo.f8E5M2>>, !vhlo.tensor<4x16x!vhlo.complex<!vhlo.f8E5M2>>) -> !vhlo.tensor<4x16x!vhlo.complex<!vhlo.f8E5M2>>
  "vhlo.return"() : () -> ()
}

// -----

// expected-error @+1 {{failed to legalize operation 'vhlo.func' that was explicitly marked illegal}}
vhlo.func @illegal_type_function(%arg0: !vhlo.f8E4M3FN) -> () {
  "vhlo.return"() : () -> ()
}

// -----

// expected-error @+1 {{failed to legalize operation 'vhlo.func' that was explicitly marked illegal}}
vhlo.func @illegal_type_function_output() -> (!vhlo.f8E4M3FN) {
  "vhlo.return"() : () -> ()
}

// -----

// expected-error @+1 {{failed to legalize operation 'vhlo.func' that was explicitly marked illegal}}
vhlo.func @illegal_type_tuple_element(%arg0: !vhlo.tuple<!vhlo.tensor<!vhlo.f32>, !vhlo.tensor<!vhlo.f8E5M2>>) -> () {
  %0 = "vhlo.get_tuple_element"(%arg0) {index = #vhlo.integer<0 : i32>} : (!vhlo.tuple<!vhlo.tensor<!vhlo.f32>, !vhlo.tensor<!vhlo.f8E5M2>>) -> !vhlo.tensor<!vhlo.f32>
  "vhlo.return"() : () -> ()
}

// -----

vhlo.func @illegal_type_quantized_storage(%arg0: !vhlo.tensor<!vhlo.f32>) -> () {
  // expected-error @+1 {{failed to legalize operation 'vhlo.uniform_quantize' that was explicitly marked illegal}}
  %0 = "vhlo.uniform_quantize"(%arg0) : (!vhlo.tensor<!vhlo.f32>) -> !vhlo.tensor<!vhlo.quant<!vhlo.f8E4M3FN:!vhlo.f32, 3.400000e+01:16, -128:127, 1>>
  "vhlo.return"() : () -> ()
}

// -----

vhlo.func @illegal_type_quantized_expressed(%arg0: !vhlo.tensor<!vhlo.f32>) -> () {
  // expected-error @+1 {{failed to legalize operation 'vhlo.uniform_quantize' that was explicitly marked illegal}}
  %0 = "vhlo.uniform_quantize"(%arg0) : (!vhlo.tensor<!vhlo.f32>) -> !vhlo.tensor<!vhlo.quant<!vhlo.i8:!vhlo.f8E5M2, 3.400000e+01:16, -128:127, 1>>
  "vhlo.return"() : () -> ()
}

// -----

vhlo.func @illegal_attr_array_element(%arg0: !vhlo.tensor<!vhlo.f32>) -> !vhlo.tensor<!vhlo.f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.custom_call_v2' that was explicitly marked illegal}}
  %0 = "vhlo.custom_call_v2"(%arg0) {
    api_version = #vhlo<api_version API_VERSION_ORIGINAL>,
    backend_config = #vhlo.string<"">,
    call_target_name = #vhlo.string<"foo">,
    called_computations = #vhlo.array<[#vhlo.float< 1.0 : !vhlo.f8E4M3FN>]>
  } : (!vhlo.tensor<!vhlo.f32>) -> !vhlo.tensor<!vhlo.f32>
  "vhlo.return"(%0) : (!vhlo.tensor<!vhlo.f32>) -> ()
}

// -----

// This simulates version validation if a new float attr kind is introduced.
vhlo.func @illegal_attr_float(%arg0: !vhlo.tensor<16x16x16x16x!vhlo.f32>, %arg1: !vhlo.tensor<16x!vhlo.f32>, %arg2: !vhlo.tensor<16x!vhlo.f32>, %arg3: !vhlo.tensor<16x!vhlo.f32>, %arg4: !vhlo.tensor<16x!vhlo.f32>) -> !vhlo.tensor<16x16x16x16x!vhlo.f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.batch_norm_inference' that was explicitly marked illegal}}
  %0 = "vhlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {
    epsilon = #vhlo.float<1.000000e-03 : !vhlo.f8E4M3FN>,
    feature_index = #vhlo.integer<0 : i64>
  } : (!vhlo.tensor<16x16x16x16x!vhlo.f32>, !vhlo.tensor<16x!vhlo.f32>, !vhlo.tensor<16x!vhlo.f32>, !vhlo.tensor<16x!vhlo.f32>, !vhlo.tensor<16x!vhlo.f32>) -> !vhlo.tensor<16x16x16x16x!vhlo.f32>
  "vhlo.return"(%0) : (!vhlo.tensor<16x16x16x16x!vhlo.f32>) -> ()
}
// This simulates version validation if a new attr is used in a dictionary.

// -----

// expected-error @+1 {{failed to legalize operation 'vhlo.func' that was explicitly marked illegal}}
vhlo.func @illegal_attr_type(%arg0: !vhlo.tensor<16x16x16x16x!vhlo.f32>) -> () {
  "vhlo.return"() : () -> ()
} {arg_attrs = #vhlo.array<[#vhlo.dict<{#vhlo.string<"stablehlo.self"> = #vhlo.type<!vhlo.f8E4M3FN>}>]>}
