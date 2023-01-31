// RUN: stablehlo-opt --vhlo-to-version='target=0.3.0' --verify-diagnostics --split-input-file %s

// This file tests that legality checks recurse into attributes and types.
// This is needed in case a new type is added to StableHLO, we need to ensure
// that these are prevented from targeting previous versions where the nested type
// or attribute is not supportend.

// NOTE: These tests can all be converted to use !vhlo.FP8* once support has been added.
// In the meantime, using an unconverted i16.

// expected-error @+2 {{unknown vhlo type: }}
// expected-error @+1 {{failed to parse VHLO_RankedTensorV1 parameter 'elementType' which is to be a `::mlir::Type`}}
func.func @verifier_illegal_type_complex_element(%arg0: !vhlo.tensor<4x16x!vhlo.complex<i16>>) -> () {
  %0 = "vhlo.add"(%arg0, %arg0) : (!vhlo.tensor<4x16x!vhlo.complex<i16>>, !vhlo.tensor<4x16x!vhlo.complex<i16>>) -> !vhlo.tensor<4x16x!vhlo.complex<i16>>
  func.return
}

// -----

// expected-error @+1 {{unknown vhlo type:}}
func.func @verifier_illegal_type_tensor_element(%arg0: !vhlo.tensor<4x16xi16>) -> () {
  %0 = "vhlo.add"(%arg0, %arg0) : (!vhlo.tensor<4x16xi16>, !vhlo.tensor<4x16xi16>) -> !vhlo.tensor<4x16xi16>
  func.return
}

// -----

// expected-error @+1 {{unknown vhlo type:}}
func.func @verifier_illegal_type_tensor_element_unranked(%arg0: !vhlo.unranked_tensor<i16>) -> () {
  %0 = "vhlo.add"(%arg0, %arg0) : (!vhlo.unranked_tensor<i16>, !vhlo.unranked_tensor<i16>) -> !vhlo.unranked_tensor<i16>
  func.return
}

// -----

// expected-error @+2 {{unknown vhlo type:}}
// expected-error @+1 {{failed to parse VHLO_TupleV1 parameter 'types' which is to be a `::llvm::ArrayRef<::mlir::Type>`}}
func.func @verifier_illegal_type_tuple_element(%arg0: !vhlo.tuple<!vhlo.tensor<!vhlo.f32>, !vhlo.tensor<i16>>) -> () {
  %0 = "vhlo.get_tuple_element"(%arg0) {index = #vhlo.integer<0 : i32>} : (!vhlo.tuple<!vhlo.tensor<!vhlo.f32>, !vhlo.tensor<i16>>) -> !vhlo.tensor<!vhlo.f32>
  func.return
}

// -----

// expected-error @+2 {{unknown vhlo type:}}
// expected-error @+1 {{failed to parse VHLO_RankedTensorV1 parameter 'elementType' which is to be a `::mlir::Type`}}
func.func @illegal_type_quantized_storage(%arg0: !vhlo.tensor<!vhlo.quant<i16:!vhlo.f32, 3.400000e+01:16, -128:127, 1>>) -> () {
  %0 = "vhlo.uniform_dequantize"(%arg0) : (!vhlo.tensor<!vhlo.quant<i16:!vhlo.f32, 3.400000e+01:16, -128:127, 1>>) -> !vhlo.tensor<!vhlo.f32>
  func.return
}

// -----

// expected-error @+2 {{unknown vhlo type:}}
// expected-error @+1 {{failed to parse VHLO_RankedTensorV1 parameter 'elementType' which is to be a `::mlir::Type`}}
func.func @verifier_illegal_type_quantized_expressed(%arg0: !vhlo.tensor<!vhlo.quant<!vhlo.i8:i16, 3.400000e+01:16, -128:127, 1>>) -> () {
  // expected-d-error @+1 {{failed to legalize operation 'vhlo.uniform_dequantize' that was explicitly marked illegal}}
  %0 = "vhlo.uniform_dequantize"(%arg0) : (!vhlo.tensor<!vhlo.quant<!vhlo.i8:i16, 3.400000e+01:16, -128:127, 1>>) -> !vhlo.tensor<!vhlo.f32>
  func.return
}
