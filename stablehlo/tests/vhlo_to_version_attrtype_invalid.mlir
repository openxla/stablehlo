// RUN: stablehlo-opt --vhlo-to-version='target=0.3.0' --verify-diagnostics --split-input-file %s

// This file tests that legality checks recurse into attributes and types.
// This is needed in case a new type is added to StableHLO, we need to ensure
// that these are prevented from targeting previous versions where the nested type
// or attribute is not supported.

// NOTE: These tests can all be converted to use !vhlo.FP8* once support has been added.
// In the meantime, using an unconverted i16.


// This simulates version validation if a new numeric type is introduced.
func.func @illegal_type_tensor_element(%arg0: !vhlo.tensor<4x16xi16>) -> () {
  // expected-error @+1 {{failed to legalize operation 'vhlo.add' that was explicitly marked illegal}}
  %0 = "vhlo.add"(%arg0, %arg0) : (!vhlo.tensor<4x16xi16>, !vhlo.tensor<4x16xi16>) -> !vhlo.tensor<4x16xi16>
  func.return
}

/// if (auto complex = type.dyn_cast<ComplexV1Type>()) {
  //   return isLegalType(complex.getElementType(), targetVersion);
  // }
  // if (auto ranked = type.dyn_cast<RankedTensorV1Type>()) {
  //   auto encoding = ranked.getEncoding();
  //   if (encoding && failed(isLegalAttribute(encoding, targetVersion)))
  //     return failure();
  //   return isLegalType(ranked.getElementType(), targetVersion);
  // }
  // if (auto quant = type.dyn_cast<UniformQuantizedV1Type>()) {
  //   return success(
  //       succeeded(isLegalType(quant.getStorageType(), targetVersion)) &&
  //       succeeded(isLegalType(quant.getExpressedType(), targetVersion)));
  // }
  // if (auto unranked = type.dyn_cast<UnrankedTensorV1Type>()) {
  //   return isLegalType(unranked.getElementType(), targetVersion);
  // }
  // if (auto tuple = type.dyn_cast<TupleV1Type>()) {
  //   return success(llvm::all_of(tuple.getTypes(), [&](Type ele) {
  //     return succeeded(isLegalType(ele, targetVersion));
  //   }));
  // }

// -----

// This following type tests simulate version validation if a new numeric type is introduced.
func.func @illegal_type_unranked_tensor_element(%arg0: !vhlo.unranked_tensor<i16>) -> () {
  // expected-error @+1 {{failed to legalize operation 'vhlo.add' that was explicitly marked illegal}}
  %0 = "vhlo.add"(%arg0, %arg0) : (!vhlo.unranked_tensor<i16>, !vhlo.unranked_tensor<i16>) -> !vhlo.unranked_tensor<i16>
  func.return
}

// -----

func.func @illegal_type_complex_element(%arg0: !vhlo.tensor<4x16x!vhlo.complex<i16>>) -> () {
  // expected-error @+1 {{failed to legalize operation 'vhlo.add' that was explicitly marked illegal}}
  %0 = "vhlo.add"(%arg0, %arg0) : (!vhlo.tensor<4x16x!vhlo.complex<i16>>, !vhlo.tensor<4x16x!vhlo.complex<i16>>) -> !vhlo.tensor<4x16x!vhlo.complex<i16>>
  func.return
}

// -----

func.func @illegal_type_tensor_element(%arg0: !vhlo.tensor<4x16xi16>) -> () {
  // expected-error @+1 {{failed to legalize operation 'vhlo.add' that was explicitly marked illegal}}
  %0 = "vhlo.add"(%arg0, %arg0) : (!vhlo.tensor<4x16xi16>, !vhlo.tensor<4x16xi16>) -> !vhlo.tensor<4x16xi16>
  func.return
}

// -----

func.func @illegal_type_tuple_element(%arg0: !vhlo.tuple<!vhlo.tensor<!vhlo.f32>, !vhlo.tensor<i16>>) -> () {
  // expected-error @+1 {{failed to legalize operation 'vhlo.get_tuple_element' that was explicitly marked illegal}}
  %0 = "vhlo.get_tuple_element"(%arg0) {index = #vhlo.integer<0 : i32>} : (!vhlo.tuple<!vhlo.tensor<!vhlo.f32>, !vhlo.tensor<i16>>) -> !vhlo.tensor<!vhlo.f32>
  func.return
}

// -----

func.func @illegal_type_quantized_storage(%arg0: !vhlo.tensor<!vhlo.quant<i16:!vhlo.f32, 3.400000e+01:16, -128:127, 1>>) -> () {
  // expected-error @+1 {{failed to legalize operation 'vhlo.uniform_dequantize' that was explicitly marked illegal}}
  %0 = "vhlo.uniform_dequantize"(%arg0) : (!vhlo.tensor<!vhlo.quant<i16:!vhlo.f32, 3.400000e+01:16, -128:127, 1>>) -> !vhlo.tensor<!vhlo.f32>
  func.return
}

// -----

func.func @illegal_type_quantized_expressed(%arg0: !vhlo.tensor<!vhlo.quant<!vhlo.integer<i8>:i16, 3.400000e+01:16, -128:127, 1>>) -> () {
  // expected-error @+1 {{failed to legalize operation 'vhlo.uniform_dequantize' that was explicitly marked illegal}}
  %0 = "vhlo.uniform_dequantize"(%arg0) : (!vhlo.tensor<!vhlo.quant<!vhlo.integer<i8>:i16, 3.400000e+01:16, -128:127, 1>>) -> !vhlo.tensor<!vhlo.f32>
  func.return
}

// -----

// This simulates version validation if a new symbol ref attr is introduced.
func.func @illegal_attr_array_element(%arg0: !vhlo.tensor<!vhlo.f32>) -> !vhlo.tensor<!vhlo.f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.custom_call_v2' that was explicitly marked illegal}}
  %0 = "vhlo.custom_call_v2"(%arg0) {
    api_version = #vhlo<api_version API_VERSION_ORIGINAL>,
    backend_config = #vhlo.string<"">,
    call_target_name = #vhlo.string<"foo">,
    called_computations = #vhlo.array<[dense<1> : tensor<i16>]>
  } : (!vhlo.tensor<!vhlo.f32>) -> !vhlo.tensor<!vhlo.f32>
  return %0 : !vhlo.tensor<!vhlo.f32>
}

// -----

// This simulates version validation if a new string attr is introduced.
func.func @vhlo_illegal_attr_symbolref(%arg0: !vhlo.tensor<!vhlo.f32>) -> !vhlo.tensor<!vhlo.f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.custom_call_v2' that was explicitly marked illegal}}
  %0 = "vhlo.custom_call_v2"(%arg0) {
    api_version = #vhlo<api_version API_VERSION_ORIGINAL>, 
    backend_config = #vhlo.string<"">, 
    call_target_name = #vhlo.string<"foo">,
    called_computations = #vhlo.array<[#vhlo.sym<dense<1> : tensor<i16>>]>
  } : (!vhlo.tensor<!vhlo.f32>) -> !vhlo.tensor<!vhlo.f32>
  return %0 : !vhlo.tensor<!vhlo.f32>
}

// -----

// This simulates version validation if a new string attr is introduced.
func.func @vhlo_illegal_attr_float(%arg0: !vhlo.tensor<16x16x16x16x!vhlo.f32>, %arg1: !vhlo.tensor<16x!vhlo.f32>, %arg2: !vhlo.tensor<16x!vhlo.f32>, %arg3: !vhlo.tensor<16x!vhlo.f32>, %arg4: !vhlo.tensor<16x!vhlo.f32>) -> !vhlo.tensor<16x16x16x16x!vhlo.f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.batch_norm_inference' that was explicitly marked illegal}}
  %0 = "vhlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {
    epsilon = #vhlo.float<1.000000e-03 : f80>, 
    feature_index = #vhlo.integer<0 : i64>
  } : (!vhlo.tensor<16x16x16x16x!vhlo.f32>, !vhlo.tensor<16x!vhlo.f32>, !vhlo.tensor<16x!vhlo.f32>, !vhlo.tensor<16x!vhlo.f32>, !vhlo.tensor<16x!vhlo.f32>) -> !vhlo.tensor<16x16x16x16x!vhlo.f32>
  return %0 : !vhlo.tensor<16x16x16x16x!vhlo.f32>
}
