// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.24.0' --vhlo-legalize-to-stablehlo %s | FileCheck %s
// RUN: not stablehlo-translate --serialize --target=1.23.0 %s 2> %t && grep -q "1.23.0" %t

// CHECK-LABEL: func @complex_roundtrip_operand_result_f16
func.func @complex_roundtrip_operand_result_f16(%arg0: tensor<4xf16>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<4xf16>) -> tensor<4xcomplex<f16>>
  // CHECK: stablehlo.convert
  // CHECK-SAME: tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @complex_roundtrip_tensor_element_f16
func.func @complex_roundtrip_tensor_element_f16(%arg0: tensor<4xcomplex<f16>>, %arg1: tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>> {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<4xcomplex<f16>>, tensor<4xcomplex<f16>>) -> tensor<4xcomplex<f16>>
  // CHECK: stablehlo.add
  // CHECK-SAME: tensor<4xcomplex<f16>>
  func.return %0 : tensor<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func @complex_roundtrip_tensor_element_bf16
func.func @complex_roundtrip_tensor_element_bf16(%arg0: tensor<4xcomplex<bf16>>, %arg1: tensor<4xcomplex<bf16>>) -> tensor<4xcomplex<bf16>> {
  %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<4xcomplex<bf16>>, tensor<4xcomplex<bf16>>) -> tensor<4xcomplex<bf16>>
  // CHECK: stablehlo.multiply
  // CHECK-SAME: tensor<4xcomplex<bf16>>
  func.return %0 : tensor<4xcomplex<bf16>>
}

// -----

// CHECK-LABEL: func @complex_roundtrip_tuple_element_f16
func.func @complex_roundtrip_tuple_element_f16(%arg0: tensor<2xf16>) -> tuple<tensor<2xcomplex<f16>>> {
  %0 = "stablehlo.custom_call"(%arg0) { call_target_name = "foo" } : (tensor<2xf16>) -> tuple<tensor<2xcomplex<f16>>>
  // CHECK: stablehlo.custom_call
  // CHECK-SAME: tuple<tensor<2xcomplex<f16>>>
  func.return %0 : tuple<tensor<2xcomplex<f16>>>
}

// -----

// CHECK-LABEL: func.func private @complex_roundtrip_func_signature_f16
func.func private @complex_roundtrip_func_signature_f16(tensor<2xcomplex<f16>>) -> tensor<2xcomplex<f16>>
// CHECK-SAME: tensor<2xcomplex<f16>>

// -----

// CHECK-LABEL: func @complex_roundtrip_type_attr_f16
func.func @complex_roundtrip_type_attr_f16() attributes { stablehlo.attr = complex<f16> } {
  return
}
// CHECK: stablehlo.attr = complex<f16>

// -----

// CHECK-LABEL: func.func private @complex_roundtrip_future_element_f16
func.func private @complex_roundtrip_future_element_f16() -> !stablehlo.future<tensor<2xcomplex<f16>>>
// CHECK-SAME: !stablehlo.future<tensor<2xcomplex<f16>>>

// -----

// CHECK-LABEL: func.func @complex_roundtrip_buffer_element_f16
func.func @complex_roundtrip_buffer_element_f16(%arg0: memref<4xcomplex<f16>>) -> memref<4xcomplex<f16>> {
  // CHECK: return %arg0 : memref<4xcomplex<f16>>
  func.return %arg0 : memref<4xcomplex<f16>>
}

// -----

// CHECK-LABEL: func.func private @complex_roundtrip_unranked_element_f16
func.func private @complex_roundtrip_unranked_element_f16(tensor<*xcomplex<f16>>) -> tensor<*xcomplex<f16>>
// CHECK-SAME: tensor<*xcomplex<f16>>
