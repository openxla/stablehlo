// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect --stablehlo-legalize-qdq-to-quantized-op | FileCheck %s --check-prefixes=CHECK

// -----

// CHECK-LABEL @compose_quantized_abs_op
// CHECK: %[[abs0:.*]] = stablehlo.abs %arg0 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT: %[[abs1:.*]] = stablehlo.abs %[[abs0]] : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT: return %[[abs1]] : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
func.func @compose_quantized_abs_op(%arg0: tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
    %0 = stablehlo.uniform_dequantize %arg0 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %1 = stablehlo.abs %0 : tensor<16x16xf32>
    %2 = stablehlo.uniform_quantize %1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %3 = stablehlo.abs %2 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    func.return %3 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL @compose_quantized_add_op
// CHECK:      %[[oper0:.*]] = stablehlo.uniform_quantize %arg0 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT: %[[oper1:.*]] = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT: %[[add_result:.*]] = stablehlo.add %[[oper0]], %[[oper1]] : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT: return %[[add_result]] : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
func.func @compose_quantized_add_op(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %1 = stablehlo.uniform_dequantize %0 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %2 = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %4 = stablehlo.add %1, %3 : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = stablehlo.uniform_quantize %4 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    func.return %5 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}
