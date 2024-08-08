// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect --stablehlo-legalize-qdq-to-quantized-op | FileCheck %s --check-prefixes=CHECK

// -----

// CHECK-LABEL @compose_quantized_abs_op
// CHECK:      %[[abs0:.*]] = stablehlo.abs %arg0 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT: return %[[abs0]] : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
func.func @compose_quantized_abs_op(%arg0: tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
    %0 = stablehlo.uniform_dequantize %arg0 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %1 = stablehlo.abs %0 : tensor<16x16xf32>
    %2 = stablehlo.uniform_quantize %1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    func.return %2 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL @failure_operand_not_defined_by_op
// CHECK-NOT: stablehlo.abs {{.*}} : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
// CHECK:     return
func.func @operand_not_defined_by_op(%arg0: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
  %1 = stablehlo.abs %arg0 : tensor<16x16xf32>
  %2 = stablehlo.uniform_quantize %1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %2 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL @failure_op_with_region
// CHECK:         %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.uniform_quantize %arg0 : (tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:    %1 = stablehlo.uniform_dequantize %0 : (tensor<4x5x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<4x5xf32>
// CHECK-NEXT:    %2 = stablehlo.uniform_quantize %arg1 : (tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:    %3 = stablehlo.uniform_dequantize %2 : (tensor<4x5x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<4x5xf32>
// CHECK-NEXT:    %4 = "stablehlo.map"(%1, %3) <{dimensions = array<i64: 0, 1>}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:      stablehlo.return %cst : tensor<f32>
// CHECK-NEXT:    }) : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
// CHECK-NEXT:    %5 = stablehlo.uniform_quantize %4 : (tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:    return %5 : tensor<4x5x!quant.uniform<u8:f32, 3.400000e+01:16>>

func.func @failure_op_with_region(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>> {
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>
  %1 = stablehlo.uniform_dequantize %0 : (tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<4x5xf32>
  %2 = stablehlo.uniform_quantize %arg1 : (tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>
  %3 = stablehlo.uniform_dequantize %2 : (tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<4x5xf32>
  %4 = "stablehlo.map"(%1, %3) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %ret = stablehlo.constant dense<2.0> : tensor<f32>
    "stablehlo.return"(%ret) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  %5 = stablehlo.uniform_quantize %4 : (tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %5 : tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL failure_varidic_op
// CHECK:       %0 = stablehlo.uniform_quantize %arg0 : (tensor<8x2xf32>) -> tensor<8x2x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  %1 = stablehlo.uniform_dequantize %0 : (tensor<8x2x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<8x2xf32>
// CHECK-NEXT:  %2 = stablehlo.uniform_quantize %arg1 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  %3 = stablehlo.uniform_dequantize %2 : (tensor<2x2x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<2x2xf32>
// CHECK-NEXT:  %4:2 = "stablehlo.all_gather"(%1, %3) {{.*}} : (tensor<8x2xf32>, tensor<2x2xf32>) -> (tensor<8x8xf32>, tensor<2x4xf32>)
// CHECK-NEXT:  %5 = stablehlo.uniform_quantize %4#0 : (tensor<8x8xf32>) -> tensor<8x8x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  return %5, %4#1 : tensor<8x8x!quant.uniform<u8:f32, 3.400000e+01:16>>, tensor<2x4xf32>
func.func @failure_varidic_op(%arg0: tensor<8x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<8x8x!quant.uniform<ui8:f32, 34.0:16>>, tensor<2x4xf32>) {
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<8x2xf32>) -> tensor<8x2x!quant.uniform<ui8:f32, 34.0:16>>
  %1 = stablehlo.uniform_dequantize %0 : (tensor<8x2x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<8x2xf32>
  %2 = stablehlo.uniform_quantize %arg1 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<ui8:f32, 34.0:16>>
  %3 = stablehlo.uniform_dequantize %2 : (tensor<2x2x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<2x2xf32>
  %4:2 = "stablehlo.all_gather"(%1, %3) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>, tensor<2x2xf32>) -> (tensor<8x8xf32>, tensor<2x4xf32>)
  %5 = stablehlo.uniform_quantize %4#0 : (tensor<8x8xf32>) -> tensor<8x8x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %5, %4#1 : tensor<8x8x!quant.uniform<ui8:f32, 34.0:16>>, tensor<2x4xf32>
}

// -----

// CHECK-LABEL @failure_all_operands_not_quantized
// CHECK:       %[[oper0:.*]] = stablehlo.uniform_quantize %arg0 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  %[[oper1:.*]] = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  %[[add_result:.*]] = stablehlo.add %[[oper0]], %[[oper1]] : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  return %[[add_result]] : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
func.func @compose_quantized_add_op(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %1 = stablehlo.uniform_dequantize %0 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %2 = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %4 = stablehlo.add %1, %3 : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = stablehlo.uniform_quantize %4 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    func.return %5 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL @failure_defining_op_is_not_a_uniform_dequantized_op
// CHECK:       %0 = stablehlo.abs %arg0 : tensor<16x16xf32>
// CHECK-NEXT:  %1 = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  %2 = stablehlo.uniform_dequantize %1 : (tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16xf32>
// CHECK-NEXT:  %3 = stablehlo.add %0, %2 : tensor<16x16xf32>
// CHECK-NEXT:  %4 = stablehlo.uniform_quantize %3 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  return %4 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
func.func @failure_defining_op_is_not_a_uniform_dequantized_op(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
    %0 = stablehlo.abs %arg0 : tensor<16x16xf32>
    %1 = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %3 = stablehlo.add %0, %2 : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = stablehlo.uniform_quantize %3 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    func.return %4: tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}
