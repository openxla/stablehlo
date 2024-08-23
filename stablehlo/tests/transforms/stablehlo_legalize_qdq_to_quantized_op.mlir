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

// CHECK-LABEL @failed_to_match_uniform_quant_op_operand_not_defined_by_op
// CHECK:       %0 = stablehlo.uniform_quantize %arg0 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  return %0 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
func.func @failed_to_match_uniform_quant_op_operand_not_defined_by_op(%arg0: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %0 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL @failed_to_match_op_with_region
// CHECK:       %0 = "stablehlo.all_reduce"(%arg0){{.*}}: tensor<1x2xi64>}> ({
// CHECK-NEXT:  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:    %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
// CHECK-NEXT:    stablehlo.return %2 : tensor<f32>
// CHECK-NEXT:    }) : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:  %1 = stablehlo.uniform_quantize %0 : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  return %1 : tensor<4x!quant.uniform<u8:f32, 3.400000e+01:16>>

func.func @failed_to_match_op_with_region(%operand0 : tensor<4xf32>) -> (tensor<4x!quant.uniform<ui8:f32, 34.0:16>>) {
  %0 = stablehlo.uniform_quantize %operand0 : (tensor<4xf32>) -> tensor<4x!quant.uniform<ui8:f32, 34.0:16>>
  %1 = stablehlo.uniform_dequantize %0 : (tensor<4x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<4xf32>
  %2 = "stablehlo.all_reduce"(%operand0) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
  }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<4xf32>) -> tensor<4xf32>
  %4 = stablehlo.uniform_quantize %2 : (tensor<4xf32>) -> tensor<4x!quant.uniform<ui8:f32, 34.0:16>>
  return %4 : tensor<4x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL failed_to_match_varidic_op
// CHECK:       %0 = stablehlo.uniform_quantize %arg0 : (tensor<8x2xf32>) -> tensor<8x2x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  %1 = stablehlo.uniform_dequantize %0 : (tensor<8x2x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<8x2xf32>
// CHECK-NEXT:  %2 = stablehlo.uniform_quantize %arg1 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  %3 = stablehlo.uniform_dequantize %2 : (tensor<2x2x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<2x2xf32>
// CHECK-NEXT:  %4:2 = "stablehlo.all_gather"(%1, %3) {{.*}} : (tensor<8x2xf32>, tensor<2x2xf32>) -> (tensor<8x8xf32>, tensor<2x4xf32>)
// CHECK-NEXT:  %5 = stablehlo.uniform_quantize %4#0 : (tensor<8x8xf32>) -> tensor<8x8x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  return %5, %4#1 : tensor<8x8x!quant.uniform<u8:f32, 3.400000e+01:16>>, tensor<2x4xf32>
func.func @failed_to_match_varidic_op(%arg0: tensor<8x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<8x8x!quant.uniform<ui8:f32, 34.0:16>>, tensor<2x4xf32>) {
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

// CHECK-LABEL @failed_to_match_operand_of_compute_op_already_quantized
// CHECK:        %0 = stablehlo.uniform_quantize %arg0 : (tensor<1x8x8x207xf32>) -> tensor<1x8x8x207x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:   %1 = stablehlo.uniform_dequantize %0 : (tensor<1x8x8x207x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<1x8x8x207xf32>
// CHECK-NEXT:   %2 = stablehlo.abs %arg1 : tensor<3x3x207x16x!quant.uniform<i8:f32, 5.000000e+00:20>>
// CHECK-NEXT:   %3 = stablehlo.convolution(%1, %2) {{.*}} : (tensor<1x8x8x207xf32>, tensor<3x3x207x16x!quant.uniform<i8:f32, 5.000000e+00:20>>) -> tensor<1x8x8x16xf32>
// CHECK-NEXT:   %4 = stablehlo.uniform_quantize %3 : (tensor<1x8x8x16xf32>) -> tensor<1x8x8x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:   return %4 : tensor<1x8x8x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
func.func @failed_to_match_operand_of_compute_op_already_quantized(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<ui8:f32, 34.0:16>> {
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<1x8x8x207xf32>) -> tensor<1x8x8x207x!quant.uniform<ui8:f32, 34.0:16>>
    %1 = stablehlo.uniform_dequantize %0 : (tensor<1x8x8x207x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<1x8x8x207xf32>
    %2 = stablehlo.abs %arg1 : tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>
    %3 = stablehlo.convolution(%1, %2)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16xf32>
    %4 = stablehlo.uniform_quantize %3 : (tensor<1x8x8x16xf32>) -> tensor<1x8x8x16x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %4 : tensor<1x8x8x16x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL @failed_to_match_operand_not_defined_by_op
// CHECK:       %0 = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  %1 = stablehlo.uniform_dequantize %0 : (tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16xf32>
// CHECK-NEXT:  %2 = stablehlo.add %arg0, %1 : tensor<16x16xf32>
// CHECK-NEXT:  %3 = stablehlo.uniform_quantize %2 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  return %3 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
func.func @failed_to_match_operand_not_defined_by_op(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
    %1 = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %3 = stablehlo.add %arg0, %2 : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = stablehlo.uniform_quantize %3 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    func.return %4: tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL @failed_to_match_defining_op_is_not_a_uniform_dequantized_op
// CHECK:       %0 = stablehlo.abs %arg0 : tensor<16x16xf32>
// CHECK-NEXT:  %1 = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  %2 = stablehlo.uniform_dequantize %1 : (tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16xf32>
// CHECK-NEXT:  %3 = stablehlo.add %0, %2 : tensor<16x16xf32>
// CHECK-NEXT:  %4 = stablehlo.uniform_quantize %3 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
// CHECK-NEXT:  return %4 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
func.func @failed_to_match_defining_op_is_not_a_uniform_dequantized_op(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
    %0 = stablehlo.abs %arg0 : tensor<16x16xf32>
    %1 = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %3 = stablehlo.add %0, %2 : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = stablehlo.uniform_quantize %3 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    func.return %4: tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}
