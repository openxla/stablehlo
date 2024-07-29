// RUN:  stablehlo-opt --stablehlo-legalize-quant-to-math -split-input-file %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @uniform_quantize_and_dequantize
func.func @uniform_quantize_and_dequantize(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[SCALES:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL0:.*]] = chlo.broadcast_divide %arg0, %[[SCALES]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL1:.*]] = chlo.broadcast_add %[[VAL0]], %[[ZPS]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL2:.*]] = stablehlo.clamp %[[QUANT_MIN]], %[[VAL1]], %[[QUANT_MAX]] : (tensor<f32>, tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = stablehlo.round_nearest_even %[[VAL2]] : tensor<?x?xf32>
  // CHECK: %[[VAL4:.*]] = stablehlo.convert %[[VAL3]] : (tensor<?x?xf32>) -> tensor<?x?xi8>
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK-DAG: %[[SCALES_DQ:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS_DQ:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL5:.*]] = stablehlo.convert %[[VAL4]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL5]], %[[ZPS_DQ]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL7:.*]] = stablehlo.convert %[[VAL6]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL8:.*]] = chlo.broadcast_multiply %[[VAL7]], %[[SCALES_DQ]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[VAL8]] : tensor<?x?xf32>
  %1 = stablehlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_convert_dequantize
func.func @uniform_quantize_convert_dequantize(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[SCALES:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL0:.*]] = chlo.broadcast_divide %arg0, %[[SCALES]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL1:.*]] = chlo.broadcast_add %[[VAL0]], %[[ZPS]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL2:.*]] = stablehlo.clamp %[[QUANT_MIN]], %[[VAL1]], %[[QUANT_MAX]] : (tensor<f32>, tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = stablehlo.round_nearest_even %[[VAL2]] : tensor<?x?xf32>
  // CHECK: %[[VAL4:.*]] = stablehlo.convert %[[VAL3]] : (tensor<?x?xf32>) -> tensor<?x?xi8>
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK: %[[VAL5:.*]] = stablehlo.bitcast_convert %[[VAL4]] : (tensor<?x?xi8>) -> tensor<?x?xi8>
  %1 = stablehlo.bitcast_convert %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xi8>

  // CHECK: %[[VAL6:.*]] = stablehlo.bitcast_convert %[[VAL5]] : (tensor<?x?xi8>) -> tensor<?x?xi8>
  %2 = stablehlo.bitcast_convert %1 : (tensor<?x?xi8>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK-DAG: %[[SCALES_DQ:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS_DQ:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL7:.*]] = stablehlo.convert %[[VAL6]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK: %[[VAL8:.*]] = chlo.broadcast_subtract %[[VAL7]], %[[ZPS_DQ]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = stablehlo.convert %[[VAL8]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL10:.*]] = chlo.broadcast_multiply %[[VAL9]], %[[SCALES_DQ]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[VAL10]] : tensor<?x?xf32>
  %3 = stablehlo.uniform_dequantize %2 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_int4
func.func @uniform_quantize_and_dequantize_int4(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[SCALES:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = stablehlo.constant dense<-8.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = stablehlo.constant dense<7.000000e+00> : tensor<f32>
  // CHECK: %[[VAL0:.*]] = chlo.broadcast_divide %arg0, %[[SCALES]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL1:.*]] = chlo.broadcast_add %[[VAL0]], %[[ZPS]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL2:.*]] = stablehlo.clamp %[[QUANT_MIN]], %[[VAL1]], %[[QUANT_MAX]] : (tensor<f32>, tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = stablehlo.round_nearest_even %[[VAL2]] : tensor<?x?xf32>
  // CHECK: %[[VAL4:.*]] = stablehlo.convert %[[VAL3]] : (tensor<?x?xf32>) -> tensor<?x?xi4>
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>

  // CHECK-DAG: %[[SCALES_DQ:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS_DQ:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL5:.*]] = stablehlo.convert %[[VAL4]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL5]], %[[ZPS_DQ]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL7:.*]] = stablehlo.convert %[[VAL6]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL8:.*]] = chlo.broadcast_multiply %[[VAL7]], %[[SCALES_DQ]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[VAL8]] : tensor<?x?xf32>
  %1 = stablehlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_type_exensions
func.func @uniform_quantize_and_dequantize_type_exensions(%arg0: tensor<?x?xf32, #stablehlo.bounds<4, 4>>) -> () {
  // CHECK: %[[QUANTIZED:.*]] = stablehlo.convert %[[VAL0:.*]] : (tensor<?x?xf32, #stablehlo.bounds<4, 4>>) -> tensor<?x?xi8, #stablehlo.bounds<4, 4>>
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<?x?xf32, #stablehlo.bounds<4, 4>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>, #stablehlo.bounds<4, 4>>
  // CHECK: %[[DEQUANTIZED:.*]] = chlo.broadcast_multiply %[[VAL1:.*]], %[[CONST_SCALE:.*]] : (tensor<?x?xf32, #stablehlo.bounds<4, 4>>, tensor<f32>) -> tensor<?x?xf32, #stablehlo.bounds<4, 4>>
  %1 = stablehlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>, #stablehlo.bounds<4, 4>>) -> tensor<?x?xf32, #stablehlo.bounds<4, 4>>
  return
}

// -----

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// CHECK: #[[$SV:.*]] = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
// CHECK-LABEL: func @uniform_quantize_and_dequantize_sparse_tensor_encoding
func.func @uniform_quantize_and_dequantize_sparse_tensor_encoding(%arg0: tensor<?xf32, #SV>) -> () {
  // CHECK: %[[QUANTIZED:.*]] = stablehlo.convert %[[VAL0:.*]] : (tensor<?xf32, #[[$SV]]>) -> tensor<?xi8, #[[$SV]]>
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<?xf32, #SV>) -> tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #SV>
  // CHECK: %[[DEQUANTIZED:.*]] = chlo.broadcast_multiply %[[VAL1:.*]], %[[CONST_SCALE:.*]] : (tensor<?xf32, #[[$SV]]>, tensor<f32>) -> tensor<?xf32, #[[$SV]]>
  %1 = stablehlo.uniform_dequantize %0 : (tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #SV>) -> tensor<?xf32, #SV>
  return
}

// -----

// CHECK-LABEL: func @quantize_per_channel
func.func @quantize_per_channel(%arg0: tensor<26x26x3x2xf32>
    ) -> tensor<26x26x3x2x!quant.uniform<i32:f32:3, {1.100000e+00:-10, 1.100000e-01:2}>> {
  // CHECK-DAG: %[[SCALES:.*]] = stablehlo.constant dense<[1.100000e+00, 1.100000e-01]>
  // CHECK-DAG: %[[ZPS:.*]] = stablehlo.constant dense<[-1.000000e+01, 2.000000e+00]>
  // CHECK-DAG: %[[QMIN:.*]] = stablehlo.constant dense<-2.14748365E+9> : tensor<f32>
  // CHECK-DAG: %[[QMAX:.*]] = stablehlo.constant dense<2.14748365E+9> : tensor<f32>
  // CHECK: %[[DIVIDE:.*]] = chlo.broadcast_divide %arg0, %[[SCALES]]
  // CHECK-SAME: {broadcast_dimensions = array<i64: 3>}
  // CHECK-SAME: (tensor<26x26x3x2xf32>, tensor<2xf32>) -> tensor<26x26x3x2xf32>
  // CHECK: %[[ADD:.*]] = chlo.broadcast_add %[[DIVIDE]], %[[ZPS]]
  // CHECK-SAME: {broadcast_dimensions = array<i64: 3>}
  // CHECK-SAME: (tensor<26x26x3x2xf32>, tensor<2xf32>) -> tensor<26x26x3x2xf32>
  // CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[QMIN]], %[[ADD]], %[[QMAX]]
  // CHECK: %[[ROUND:.*]] = stablehlo.round_nearest_even %[[CLAMP]]
  // CHECK: %[[RESULT:.*]] = stablehlo.convert %[[ROUND]]
  // CHECK-SAME: (tensor<26x26x3x2xf32>) -> tensor<26x26x3x2xi32>
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<26x26x3x2xf32>
      ) -> tensor<26x26x3x2x!quant.uniform<i32:f32:3, {1.100000e+00:-10, 1.100000e-01:2}>>
  return %0 : tensor<26x26x3x2x!quant.uniform<i32:f32:3, {1.100000e+00:-10, 1.100000e-01:2}>>
}

// -----

// CHECK-LABEL: func @dequantize_per_channel
func.func @dequantize_per_channel(
    %arg0: tensor<26x26x3x2x!quant.uniform<i32:f32:3, {1.100000e+00:-10, 1.100000e-01:2}>>
  ) -> tensor<26x26x3x2xf32> {
  // CHECK-DAG: %[[SCALES:.*]] = stablehlo.constant dense<[1.100000e+00, 1.100000e-01]>
  // CHECK-DAG: %[[ZPS:.*]] = stablehlo.constant dense<[-10, 2]> : tensor<2xi32>
  // CHECK: %[[SUBTRACT:.*]] = chlo.broadcast_subtract
  // CHECK-SAME: %[[INPUT:.*]], %[[ZPS]]
  // CHECK-SAME: {broadcast_dimensions = array<i64: 3>}
  // CHECK-SAME: (tensor<26x26x3x2xi32>, tensor<2xi32>) -> tensor<26x26x3x2xi32>
  // CHECK: %[[FLOAT:.*]] = stablehlo.convert %[[SUBTRACT]]
  // CHECK: %[[RESULT:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[FLOAT]], %[[SCALES]]
  // CHECK-SAME: {broadcast_dimensions = array<i64: 3>}
  // CHECK-SAME: (tensor<26x26x3x2xf32>, tensor<2xf32>) -> tensor<26x26x3x2xf32>
  %0 = stablehlo.uniform_dequantize %arg0 : (
      tensor<26x26x3x2x!quant.uniform<i32:f32:3, {1.100000e+00:-10, 1.100000e-01:2}>>
    ) -> tensor<26x26x3x2xf32>
  return %0 : tensor<26x26x3x2xf32>
}

// -----

// CHECK-LABEL: func @add
func.func @add(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>> {
  // CHECK: %[[VAL1:.*]] = stablehlo.convert %[[VAL0:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK: %[[VAL3:.*]] = stablehlo.convert %[[VAL2:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL1]], %[[VAL3]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL4]], %[[VAL5]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = stablehlo.clamp %[[VAL7:.*]], %[[VAL6]], %[[VAL8:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = stablehlo.convert %[[VAL9]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %0 = stablehlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>,
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return %0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @add_i32
func.func @add_i32(
    %arg0: tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>,
    %arg1: tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: %[[VAL1:.*]] = stablehlo.convert %[[VAL0:.*]] : tensor<?x?xi32>
  // CHECK: %[[VAL3:.*]] = stablehlo.convert %[[VAL2:.*]] : tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL1:.*]], %[[VAL3:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL4]], %[[VAL5]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-NEXT: return
  %2 = stablehlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>,
      tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
    ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %2 : tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @add_int4
func.func @add_int4(
    %arg0: tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>,
    %arg1: tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  ) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>> {
  // CHECK: %[[VAL1:.*]] = stablehlo.convert %[[VAL0:.*]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK: %[[VAL3:.*]] = stablehlo.convert %[[VAL2:.*]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL1]], %[[VAL3]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL4]], %[[VAL5]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = stablehlo.clamp %[[VAL7:.*]], %[[VAL6]], %[[VAL8:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = stablehlo.convert %[[VAL9]] : (tensor<?x?xi32>) -> tensor<?x?xi4>
  %0 = stablehlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>,
      tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
    ) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  return %0 : tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: @add_different_lhs_type
func.func @add_different_lhs_type(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>> {
  // CHECK-DAG: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[LHS:.*]] = stablehlo.convert %arg0 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[MUL:.*]] = chlo.broadcast_multiply %[[LHS]], %[[COMBINED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[COMBINED_ZP:.*]] = stablehlo.constant dense<-5.000000e+00>
  // CHECK: %[[LHS_32:.*]] = chlo.broadcast_add %[[MUL]], %[[COMBINED_ZP]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>

  // CHECK-DAG: %[[RHS_32:.*]] = stablehlo.convert %[[RHS:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[RES_ZPS:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: %[[VAL7:.*]] = chlo.broadcast_add %[[LHS_32_REQ:.*]], %[[RHS_32:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL9:.*]] = chlo.broadcast_subtract %[[VAL7:.*]], %[[RES_ZPS:.*]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = stablehlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = stablehlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL10:.*]] = stablehlo.clamp %[[QUANT_MIN:.*]], %[[VAL9:.*]], %[[QUANT_MAX:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL11:.*]] = stablehlo.convert %[[VAL10:.*]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %2 = stablehlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>,
      tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return %2 : tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
}

// -----

// CHECK-LABEL: @add_different_rhs_type
func.func @add_different_rhs_type(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>> {
  // CHECK-DAG: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[RHS:.*]] = stablehlo.convert %arg1 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[MUL:.*]] = chlo.broadcast_multiply %[[RHS]], %[[COMBINED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[COMBINED_ZP:.*]] = stablehlo.constant dense<-5.000000e+00>
  // CHECK: %[[RHS_32:.*]] = chlo.broadcast_add %[[MUL]], %[[COMBINED_ZP]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>

  // CHECK-DAG: %[[RES_ZPS:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: %[[VAL7:.*]] = chlo.broadcast_add %[[LHS_32:.*]], %[[RHS_32_REQ:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL9:.*]] = chlo.broadcast_subtract %[[VAL7:.*]], %[[RES_ZPS:.*]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = stablehlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = stablehlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL10:.*]] = stablehlo.clamp %[[QUANT_MIN:.*]], %[[VAL9:.*]], %[[QUANT_MAX:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL11:.*]] = stablehlo.convert %[[VAL10:.*]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %0 = stablehlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>,
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return %0 : tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
}

// CHECK-LABEL: @add_different_res_type
func.func @add_different_res_type(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>> {
  // CHECK-DAG: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[LHS:.*]] = stablehlo.convert %arg0 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[MUL:.*]] = chlo.broadcast_multiply %[[LHS]], %[[COMBINED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[COMBINED_ZP:.*]] = stablehlo.constant dense<-5.000000e+00>
  // CHECK: %[[LHS_32_REQ:.*]] = chlo.broadcast_add %[[MUL]], %[[COMBINED_ZP]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>

  // CHECK-DAG: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[RHS:.*]] = stablehlo.convert %arg1 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[MUL:.*]] = chlo.broadcast_multiply %[[RHS]], %[[COMBINED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[COMBINED_ZP:.*]] = stablehlo.constant dense<-5.000000e+00>
  // CHECK: %[[RHS_32_REQ:.*]] = chlo.broadcast_add %[[MUL]], %[[COMBINED_ZP]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>

  // CHECK-DAG: %[[RES_ZPS:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: %[[VAL11:.*]] = chlo.broadcast_add %[[LHS_32_REQ:.*]], %[[RHS_32_REQ:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL12:.*]] = chlo.broadcast_subtract %[[VAL11:.*]], %[[RES_ZPS:.*]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = stablehlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = stablehlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL13:.*]] = stablehlo.clamp %[[QUANT_MIN:.*]], %[[VAL12:.*]], %[[QUANT_MAX:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL14:.*]] = stablehlo.convert %[[VAL13:.*]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %0 = stablehlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>,
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return %0 : tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
}

// -----

// CHECK-LABEL: func @add_per_channel
func.func @add_per_channel(
    %arg0: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5:3,5.8952903030815205E-5:2}>>,
    %arg1: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5:3,5.8952903030815205E-5:2}>>
  ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5:3,5.8952903030815205E-5:2}>> {
  // CHECK: %[[ADD:.*]] = stablehlo.add {{.*}} : tensor<?x3x4x2xi32>
  // CHECK: %[[ZPS:.*]] = stablehlo.constant dense<[3, 2]> : tensor<2xi32>
  // CHECK: %[[BCAST_SUB:.*]] = chlo.broadcast_subtract %[[ADD]], %[[ZPS]]
  // CHECK-SAME: {broadcast_dimensions = array<i64: 3>}
  // CHECK-SAME: (tensor<?x3x4x2xi32>, tensor<2xi32>) -> tensor<?x3x4x2xi32>
  // CHECK: return %[[BCAST_SUB]] : tensor<?x3x4x2xi32>
  %11 = stablehlo.add %arg0, %arg1 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5:3,5.8952903030815205E-5:2}>>
  return %11 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5:3,5.8952903030815205E-5:2}>>
}

// -----

// CHECK-LABEL: func @add_per_channel_no_zp
func.func @add_per_channel_no_zp(
    %arg0: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
    %arg1: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>> {
  // CHECK: %[[ADD:.*]] = stablehlo.add {{.*}} : tensor<?x3x4x2xi32>
  // CHECK: return %[[ADD]] : tensor<?x3x4x2xi32>
  %11 = stablehlo.add %arg0, %arg1 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  return %11 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
}

// -----

// CHECK-LABEL: func.func @add_per_channel_i8
func.func @add_per_channel_i8(
    %arg0: tensor<?x3x4x2x!quant.uniform<i8:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
    %arg1: tensor<?x3x4x2x!quant.uniform<i8:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  ) -> tensor<?x3x4x2x!quant.uniform<i8:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>> {
  // CHECK: stablehlo.add {{.*}} : tensor<?x3x4x2xf32>
  %11 = stablehlo.add %arg0, %arg1 : tensor<?x3x4x2x!quant.uniform<i8:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  return %11 : tensor<?x3x4x2x!quant.uniform<i8:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
}

// -----

// CHECK-LABEL: func.func @add_per_channel_different_quant_types
func.func @add_per_channel_different_quant_types(
    %arg0: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
    %arg1: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {1.1:2,0.4:-3}>>
  ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>> {
  // CHECK: stablehlo.add {{.*}} : tensor<?x3x4x2xf32>
  %11 = stablehlo.add %arg0, %arg1 : (
      tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
      tensor<?x3x4x2x!quant.uniform<i32:f32:3, {1.1:2,0.4:-3}>>
    ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  return %11 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
}

// -----

// CHECK-LABEL: func.func @add_per_channel_per_tensor_mix
func.func @add_per_channel_per_tensor_mix(
    %arg0: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
    %arg1: tensor<?x3x4x2x!quant.uniform<i32:f32, 1.1:2>>
  ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>> {
  // CHECK: stablehlo.add {{.*}} : tensor<?x3x4x2xf32>
  %11 = stablehlo.add %arg0, %arg1 : (
      tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
      tensor<?x3x4x2x!quant.uniform<i32:f32, 1.1:2>>
    ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  return %11 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
}

// -----

// CHECK-LABEL: func @requantize
func.func @requantize(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>> {
  // CHECK-DAG: %[[MERGED_ZP:.*]] = stablehlo.constant dense<-5.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[MERGED_SCALE:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[VAL1:.*]] = stablehlo.convert %arg0 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[VAL2:.*]] = chlo.broadcast_multiply %[[VAL1]], %[[MERGED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_add %[[VAL2]], %[[MERGED_ZP]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL4:.*]] = stablehlo.clamp %[[QUANT_MIN]], %[[VAL3]], %[[QUANT_MAX]] : (tensor<f32>, tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL5:.*]] = stablehlo.round_nearest_even %[[VAL4]] : tensor<?x?xf32>
  // CHECK: %[[VAL6:.*]] = stablehlo.convert %[[VAL5]] : (tensor<?x?xf32>) -> tensor<?x?xi8>
  %0 = stablehlo.uniform_quantize %arg0 : (
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return %0 : tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
}

// -----

// CHECK-LABEL: func @requantize_merged_zp_zero
func.func @requantize_merged_zp_zero(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00>> {
  // CHECK-DAG: %[[MERGED_SCALE:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[VAL1:.*]] = stablehlo.convert %arg0 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_multiply %[[VAL1]], %[[MERGED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL3:.*]] = stablehlo.clamp %[[QUANT_MIN]], %[[VAL2]], %[[QUANT_MAX]] : (tensor<f32>, tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL4:.*]] = stablehlo.round_nearest_even %[[VAL3]] : tensor<?x?xf32>
  // CHECK: %[[VAL5:.*]] = stablehlo.convert %[[VAL4]] : (tensor<?x?xf32>) -> tensor<?x?xi8>
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01>>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00>>
  return %0 : tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00>>
}

// -----

// CHECK-LABEL: func @requantize_per_channel
func.func @requantize_per_channel(
    %arg0: tensor<2x2x!quant.uniform<i8:f32:1, {1.000000e+01:3, 5.000000e+00:2}>>
  ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>> {
  // CHECK-DAG: %[[VAL1:.*]] = stablehlo.convert %arg0 : (tensor<2x2xi8>) -> tensor<2x2xf32>
  // CHECK-DAG: %[[MERGED_SCALE:.*]] = stablehlo.constant dense<[2.000000e+00, 5.000000e-01]> : tensor<2xf32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_multiply %[[VAL1]], %[[MERGED_SCALE]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[MERGED_ZP:.*]] = stablehlo.constant dense<[-5.000000e+00, -2.000000e+00]> : tensor<2xf32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_add %[[VAL2]], %[[MERGED_ZP]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL4:.*]] = stablehlo.clamp %[[QUANT_MIN]], %[[VAL3]], %[[QUANT_MAX]]
  // CHECK: %[[VAL5:.*]] = stablehlo.round_nearest_even %[[VAL4]] : tensor<2x2xf32>
  // CHECK: %[[VAL6:.*]] = stablehlo.convert %[[VAL5]] : (tensor<2x2xf32>) -> tensor<2x2xi8>
  %0 = stablehlo.uniform_quantize %arg0 : (
      tensor<2x2x!quant.uniform<i8:f32:1, {1.000000e+01:3, 5.000000e+00:2}>>
    ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
}

// -----

// CHECK-LABEL: func @requantize_per_channel_to_per_tensor
func.func @requantize_per_channel_to_per_tensor(
    %arg0: tensor<2x2x!quant.uniform<i8:f32:1, {1.000000e+01:3, 5.000000e+00:2}>>
  ) -> tensor<2x2x!quant.uniform<i8:f32, 5.000000e+00:1>> {
  // CHECK-DAG: %[[VAL1:.*]] = stablehlo.convert %arg0 : (tensor<2x2xi8>) -> tensor<2x2xf32>
  // CHECK-DAG: %[[MERGED_SCALE:.*]] = stablehlo.constant dense<[2.000000e+00, 1.000000e+00]> : tensor<2xf32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_multiply %[[VAL1]], %[[MERGED_SCALE]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[MERGED_ZP:.*]] = stablehlo.constant dense<[-5.000000e+00, -1.000000e+00]> : tensor<2xf32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_add %[[VAL2]], %[[MERGED_ZP]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL4:.*]] = stablehlo.clamp %[[QUANT_MIN]], %[[VAL3]], %[[QUANT_MAX]]
  // CHECK: %[[VAL5:.*]] = stablehlo.round_nearest_even %[[VAL4]] : tensor<2x2xf32>
  // CHECK: %[[VAL6:.*]] = stablehlo.convert %[[VAL5]] : (tensor<2x2xf32>) -> tensor<2x2xi8>
  %0 = stablehlo.uniform_quantize %arg0 : (
      tensor<2x2x!quant.uniform<i8:f32:1, {1.000000e+01:3, 5.000000e+00:2}>>
    ) -> tensor<2x2x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 5.000000e+00:1>>
}

// -----

// CHECK-LABEL: func @requantize_per_tensor_to_per_channel
func.func @requantize_per_tensor_to_per_channel(
    %arg0: tensor<2x2x!quant.uniform<i8:f32, 5.000000e+00:2>>
  ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>> {
  // CHECK-DAG: %[[VAL1:.*]] = stablehlo.convert %arg0 : (tensor<2x2xi8>) -> tensor<2x2xf32>
  // CHECK-DAG: %[[MERGED_SCALE:.*]] = stablehlo.constant dense<[1.000000e+00, 5.000000e-01]> : tensor<2xf32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_multiply %[[VAL1]], %[[MERGED_SCALE]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[MERGED_ZP:.*]] = stablehlo.constant dense<[-1.000000e+00, -2.000000e+00]> : tensor<2xf32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_add %[[VAL2]], %[[MERGED_ZP]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL4:.*]] = stablehlo.clamp %[[QUANT_MIN]], %[[VAL3]], %[[QUANT_MAX]]
  // CHECK: %[[VAL5:.*]] = stablehlo.round_nearest_even %[[VAL4]] : tensor<2x2xf32>
  // CHECK: %[[VAL6:.*]] = stablehlo.convert %[[VAL5]] : (tensor<2x2xf32>) -> tensor<2x2xi8>
  %0 = stablehlo.uniform_quantize %arg0 : (
      tensor<2x2x!quant.uniform<i8:f32, 5.000000e+00:2>>
    ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
}

// -----

func.func @requantize_per_channel_change_axis(
    %arg0: tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+01:3, 5.000000e+00:2}>>
  ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>> {
  // expected-error@+2 {{Cannot requantize while changing quantization_axis}}
  // expected-error@+1 {{failed to legalize operation 'stablehlo.uniform_quantize' that was explicitly marked illegal}}
  %0 = stablehlo.uniform_quantize %arg0 : (
      tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+01:3, 5.000000e+00:2}>>
    ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
}

// -----

// CHECK-LABEL: func @dot
func.func @dot(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
               %arg1: tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: stablehlo.dot_general
  // CHECK-SAME: contracting_dims = [1] x [0]
  // CHECK-SAME: (tensor<2x2xi8>, tensor<2x2xi8>) -> tensor<2x2xi32>
  %0 = "stablehlo.dot" (%arg0, %arg1) : (
      tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_int4
func.func @dot_int4(
    %arg0: tensor<2x2x!quant.uniform<i4:f32, 1.000000e+00:3>>,
    %arg1: tensor<2x2x!quant.uniform<i4:f32, 1.000000e+00:3>>
  ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: stablehlo.dot_general
  // CHECK-SAME: contracting_dims = [1] x [0]
  // CHECK-SAME: (tensor<2x2xi4>, tensor<2x2xi4>) -> tensor<2x2xi32>
  %0 = "stablehlo.dot" (%arg0, %arg1): (
      tensor<2x2x!quant.uniform<i4:f32, 1.000000e+00:3>>,
      tensor<2x2x!quant.uniform<i4:f32, 1.000000e+00:3>>
    ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_dynamic
func.func @dot_dynamic(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:2>>
  ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: %[[DOT:.*]] = stablehlo.dot_general
  // CHECK-SAME: contracting_dims = [1] x [0]
  // CHECK-SAME: (tensor<?x?xi8>, tensor<?x?xi8>) -> tensor<?x?xi32>

  // CHECK: stablehlo.reduce
  // CHECK-SAME: applies stablehlo.add across dimensions = [1]
  // CHECK-SAME: (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: stablehlo.get_dimension_size %[[DOT]]
  // CHECK-SAME: dim = 0 : (tensor<?x?xi32>) -> tensor<i32>
  // CHECK: stablehlo.get_dimension_size %[[DOT]]
  // CHECK-SAME: dim = 1 : (tensor<?x?xi32>) -> tensor<i32>
  // CHECK: %[[DYN_DIMS:.*]] = stablehlo.concatenate
  // CHECK-SAME: dim = 0
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK-SAME: %[[DYN_DIMS]]
  // CHECK-SAME: dims = [0]
  // CHECK-SAME: (tensor<?xi32>, tensor<2xi64>) -> tensor<?x?xi32>

  // CHECK: stablehlo.reduce
  // CHECK-SAME: applies stablehlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK-SAME: %[[DYN_DIMS]]
  // CHECK-SAME: dims = [1]
  // CHECK-SAME: (tensor<?xi32>, tensor<2xi64>) -> tensor<?x?xi32>
  %0 = "stablehlo.dot" (%arg0, %arg1) : (
      tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:2>>
    ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_dynamic_int4
func.func @dot_dynamic_int4(
    %arg0: tensor<?x?x!quant.uniform<i4:f32, 2.000000e+00:3>>,
    %arg1: tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:2>>
  ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: stablehlo.dot_general
  // CHECK-SAME: contracting_dims = [1] x [0]
  // CHECK-SAME: (tensor<?x?xi4>, tensor<?x?xi4>) -> tensor<?x?xi32>
  %0 = "stablehlo.dot" (%arg0, %arg1) : (
      tensor<?x?x!quant.uniform<i4:f32, 2.000000e+00:3>>,
      tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:2>>
    ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_dynamic_contracting_dim
func.func @dot_dynamic_contracting_dim(
    %arg0: tensor<2x?x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<?x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: stablehlo.dot_general
  // CHECK-SAME: contracting_dims = [1] x [0]
  // CHECK-SAME: (tensor<2x?xi8>, tensor<?x2xi8>) -> tensor<2x2xi32>

  // CHECK: stablehlo.reduce
  // CHECK-SAME: applies stablehlo.add across dimensions = [1]
  // CHECK-SAME: (tensor<2x?xi32>, tensor<i32>) -> tensor<2xi32>

  // CHECK: stablehlo.reduce
  // CHECK-SAME: applies stablehlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<?x2xi32>, tensor<i32>) -> tensor<2xi32>

  // CHECK: %[[DYNAMIC_DIM_INIT:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK: %[[DYNAMIC_DIM:.*]] = stablehlo.get_dimension_size
  // CHECK-SAME: dim = 0 : (tensor<?x2xi8>) -> tensor<i32>
  // CHECK: %[[DYNAMIC_DIM_TOTAL:.*]] = stablehlo.multiply
  // CHECK-SAME: %[[DYNAMIC_DIM_INIT]], %[[DYNAMIC_DIM]]
  // CHECK: %[[DIMS:.*]] = stablehlo.constant dense<9> : tensor<i32>
  // CHECK: %[[DIMS_1:.*]] = stablehlo.multiply %[[DIMS]], %[[DYNAMIC_DIM_TOTAL]]
  // CHECK: chlo.broadcast_subtract %[[ZP_OFFSET:.*]], %[[DIMS:.*]]
  %0 = "stablehlo.dot" (%arg0, %arg1) : (
      tensor<2x?x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<?x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_dynamic_result_dim
func.func @dot_dynamic_result_dim(
    %arg0: tensor<?x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<2x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: stablehlo.dot_general
  // CHECK-SAME: contracting_dims = [1] x [0]
  // CHECK-SAME: (tensor<?x2xi8>, tensor<2x?xi8>) -> tensor<?x?xi32>

  // CHECK: stablehlo.reduce
  // CHECK-SAME: applies stablehlo.add across dimensions = [1]
  // CHECK-SAME: (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK-SAME: dims = [0]
  // CHECK-SAME: (tensor<?xi32>, tensor<2xi64>) -> tensor<?x?xi32>

  // CHECK: stablehlo.reduce
  // CHECK-SAME: applies stablehlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<2x?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK-SAME: dims = [1]
  // CHECK-SAME: (tensor<?xi32>, tensor<2xi64>) -> tensor<?x?xi32>

  %0 = "stablehlo.dot" (%arg0, %arg1) : (
      tensor<?x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<2x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_dynamic_batch_dim
func.func @dot_dynamic_batch_dim(
    %arg0: tensor<?x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  ) -> tensor<?x2x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: stablehlo.dot_general
  // CHECK-SAME: contracting_dims = [1] x [0]
  // CHECK-SAME: (tensor<?x2xi8>, tensor<2x2xi8>) -> tensor<?x2xi32>

  // CHECK: stablehlo.reduce
  // CHECK-SAME: applies stablehlo.add across dimensions = [1]
  // CHECK-SAME: (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK-SAME: dims = [0]
  // CHECK-SAME: (tensor<?xi32>, tensor<2xi64>) -> tensor<?x2xi32>

  // CHECK: stablehlo.reduce
  // CHECK-SAME: applies stablehlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<2x2xi32>, tensor<i32>) -> tensor<2xi32>
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK-SAME: dims = [1]
  // CHECK-SAME: (tensor<2xi32>, tensor<2xi64>) -> tensor<?x2xi32>

  %0 = "stablehlo.dot" (%arg0, %arg1) : (
      tensor<?x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<?x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<?x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_general
func.func @dot_general(
    %arg0: tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<6x8x2x!quant.uniform<i8:f32, 1.000000e+00:0>>
  ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = stablehlo.dot_general
  // CHECK-SAME: batching_dims = [0] x [2]
  // CHECK-SAME: contracting_dims = [2] x [0]

  // Zero point offset contribution from LHS tensor * RHS ZP is 0 and skipped.

  // Zero point offset contribution from RHS tensor * LHS ZP.

  // CHECK: %[[RHS_I32:.*]] = stablehlo.convert %[[RHS:.*]] : (tensor<6x8x2xi8>)
  // CHECK-SAME: -> tensor<6x8x2xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = stablehlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies stablehlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<6x8x2xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<8x2xi32>, tensor<i32>) -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = stablehlo.broadcast_in_dim %[[RHS_ZP_CONTRIB]]
  // CHECK-SAME: dims = [2, 0]
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<2x5x8xi32>

  // Zero point offset contribution from LHS ZP * RHS ZP is 0 and skipped.

  // Combine dot result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = stablehlo.convert %[[DOT_RES]]
  // CHECK-SAME: (tensor<2x5x8xi32>) -> tensor<2x5x8xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = stablehlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<2x5x8xf32>) -> tensor<2x5x8xi32>

  // CHECK: %[[ZP_TOTAL_1:.*]] = stablehlo.convert %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<2x5x8xi32>) -> tensor<2x5x8xf32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_1:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_3:.*]] = stablehlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<2x5x8xf32>) -> tensor<2x5x8xi32>

  // CHECK: %[[RES_ZP:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_3]]
  // CHECK-SAME: (tensor<i32>, tensor<2x5x8xi32>) -> tensor<2x5x8xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_4]]

  %0 = "stablehlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<6x8x2x!quant.uniform<i8:f32, 1.000000e+00:0>>
    ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
  return %0 : tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @dot_general_combined_scale_1
func.func @dot_general_combined_scale_1(
    %arg0: tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<6x8x2x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<2x5x8x!quant.uniform<i32:f32, 6.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = stablehlo.dot_general
  // CHECK-SAME: batching_dims = [0] x [2]
  // CHECK-SAME: contracting_dims = [2] x [0]

  // Zero point offset contribution from LHS tensor * RHS ZP is 0 and skipped.

  // Zero point offset contribution from RHS tensor * LHS ZP.

  // CHECK: %[[RHS_I32:.*]] = stablehlo.convert %[[RHS:.*]] : (tensor<6x8x2xi8>)
  // CHECK-SAME: -> tensor<6x8x2xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = stablehlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies stablehlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<6x8x2xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<8x2xi32>, tensor<i32>) -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = stablehlo.broadcast_in_dim %[[RHS_ZP_CONTRIB]]
  // CHECK-SAME: dims = [2, 0]
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<2x5x8xi32>

  // CHECK: %[[RES_ZP:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_1:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<i32>, tensor<2x5x8xi32>) -> tensor<2x5x8xi32>
  // CHECK: chlo.broadcast_add %[[DOT_RES]], %[[ZP_TOTAL_1]]

  %0 = "stablehlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<6x8x2x!quant.uniform<i8:f32, 3.000000e+00:0>>
    ) -> tensor<2x5x8x!quant.uniform<i32:f32, 6.000000e+00:7>>
  return %0 : tensor<2x5x8x!quant.uniform<i32:f32, 6.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @dot_general_multiple_batching_dims
func.func @dot_general_multiple_batching_dims(
    %arg0: tensor<2x5x3x7x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<6x2x7x8x3x!quant.uniform<i8:f32, 1.000000e+00:0>>
  ) -> tensor<2x3x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = stablehlo.dot_general
  // CHECK-SAME: batching_dims = [0, 2] x [1, 4]
  // CHECK-SAME: contracting_dims = [4, 3] x [0, 2]


  // Zero point offset contribution from RHS tensor * LHS ZP.

  // CHECK: %[[RHS_I32:.*]] = stablehlo.convert %[[RHS:.*]] : (tensor<6x2x7x8x3xi8>)
  // CHECK-SAME: -> tensor<6x2x7x8x3xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = stablehlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies stablehlo.add across dimensions = [0, 2]
  // CHECK-SAME: (tensor<6x2x7x8x3xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<2x8x3xi32>
  // CHECK: %[[RHS_ZP:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<2x8x3xi32>, tensor<i32>) -> tensor<2x8x3xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = stablehlo.broadcast_in_dim %[[RHS_ZP_CONTRIB]]
  // CHECK-SAME: dims = [0, 3, 1]
  // CHECK-SAME: (tensor<2x8x3xi32>) -> tensor<2x3x5x8xi32>


  // Combine dot result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = stablehlo.convert %[[DOT_RES]]
  // CHECK-SAME: (tensor<2x3x5x8xi32>) -> tensor<2x3x5x8xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = stablehlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<2x3x5x8xf32>) -> tensor<2x3x5x8xi32>

  // CHECK: %[[ZP_TOTAL_1:.*]] = stablehlo.convert %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<2x3x5x8xi32>) -> tensor<2x3x5x8xf32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_1:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_3:.*]] = stablehlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<2x3x5x8xf32>) -> tensor<2x3x5x8xi32>

  // CHECK: %[[RES_ZP:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_3]]
  // CHECK-SAME: (tensor<i32>, tensor<2x3x5x8xi32>) -> tensor<2x3x5x8xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_4]]

  %0 = "stablehlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 2],
      rhs_batching_dimensions = [1, 4],
      lhs_contracting_dimensions = [4, 3],
      rhs_contracting_dimensions = [0, 2]
    >} : (
      tensor<2x5x3x7x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<6x2x7x8x3x!quant.uniform<i8:f32, 1.000000e+00:0>>
    ) -> tensor<2x3x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
  return %0 : tensor<2x3x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @dot_general_rhs_zero_zp
func.func @dot_general_rhs_zero_zp(
    %arg0: tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<6x8x2x!quant.uniform<i8:f32, 1.000000e+00:0>>
  ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = stablehlo.dot_general
  // CHECK-SAME: batching_dims = [0] x [2]
  // CHECK-SAME: contracting_dims = [2] x [0]

  // Zero point offset contribution from LHS tensor * RHS ZP is 0 and skipped.

  // Zero point offset contribution from RHS tensor * LHS ZP.

  // CHECK: %[[RHS_I32:.*]] = stablehlo.convert %[[RHS:.*]] : (tensor<6x8x2xi8>)
  // CHECK-SAME: -> tensor<6x8x2xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = stablehlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies stablehlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<6x8x2xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<8x2xi32>, tensor<i32>) -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = stablehlo.broadcast_in_dim %[[RHS_ZP_CONTRIB]]
  // CHECK-SAME: dims = [2, 0]
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<2x5x8xi32>

  // Zero point offset contribution from LHS ZP * RHS ZP is 0 and skipped.

  // Combine dot result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = stablehlo.convert %[[DOT_RES]]
  // CHECK-SAME: (tensor<2x5x8xi32>) -> tensor<2x5x8xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = stablehlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<2x5x8xf32>) -> tensor<2x5x8xi32>

  // CHECK: %[[ZP_TOTAL_1:.*]] = stablehlo.convert %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<2x5x8xi32>) -> tensor<2x5x8xf32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_1:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_3:.*]] = stablehlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<2x5x8xf32>) -> tensor<2x5x8xi32>

  // CHECK: %[[RES_ZP:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_3]]
  // CHECK-SAME: (tensor<i32>, tensor<2x5x8xi32>) -> tensor<2x5x8xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_4]]

  %0 = "stablehlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<6x8x2x!quant.uniform<i8:f32, 1.000000e+00:0>>
    ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
  return %0 : tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @dot_general_zero_zp
func.func @dot_general_zero_zp(
    %arg0: tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:0>>,
    %arg1: tensor<6x8x2x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = stablehlo.dot_general
  // CHECK-SAME: batching_dims = [0] x [2]
  // CHECK-SAME: contracting_dims = [2] x [0]

  // Both LHS/RHS have zero zp. No zp contribution.

  // CHECK-DAG: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<1.500000e+00> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = stablehlo.convert %[[DOT_RES]] :
  // CHECK-SAME: (tensor<2x5x8xi32>) -> tensor<2x5x8xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = stablehlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<2x5x8xf32>) -> tensor<2x5x8xi32>

  // CHECK: %[[RES_ZP:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[RES_ZP]]

  %0 = "stablehlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:0>>,
      tensor<6x8x2x!quant.uniform<i8:f32, 3.000000e+00:0>>
    ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
  return %0 : tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @dot_general_multiple_dynamic_dims
func.func @dot_general_multiple_dynamic_dims(
    %arg0: tensor<?x?x3x?x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<6x?x?x8x3x!quant.uniform<i8:f32, 1.000000e+00:0>>
  ) -> tensor<?x3x?x8x!quant.uniform<i32:f32, 4.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = stablehlo.dot_general
  // CHECK-SAME: batching_dims = [0, 2] x [1, 4]
  // CHECK-SAME: contracting_dims = [4, 3] x [0, 2]

  // Zero point offset contribution from LHS tensor * RHS ZP.

  // CHECK: %[[RHS_I32:.*]] = stablehlo.convert %[[RHS:.*]] : (tensor<6x?x?x8x3xi8>)
  // CHECK-SAME: -> tensor<6x?x?x8x3xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = stablehlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies stablehlo.add across dimensions = [0, 2]
  // CHECK-SAME: (tensor<6x?x?x8x3xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<?x8x3xi32>
  // CHECK: %[[RHS_ZP:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<?x8x3xi32>, tensor<i32>) -> tensor<?x8x3xi32>

  // Calculate output dynamic dims.
  // CHECK: %[[DIM_1_1:.*]] = stablehlo.get_dimension_size %[[DOT_RES]]
  // CHECK-SAME: dim = 0
  // CHECK: %[[DIM_1_2:.*]] = stablehlo.convert %[[DIM_1_1]] : (tensor<i32>) -> tensor<i64>
  // CHECK: %[[DIM_1:.*]] = stablehlo.reshape %[[DIM_1_2]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[DIM_2:.*]] = stablehlo.constant dense<3> : tensor<1xi64>
  // CHECK: %[[DIM_3_1:.*]] = stablehlo.get_dimension_size %[[DOT_RES]]
  // CHECK-SAME: dim = 2
  // CHECK: %[[DIM_3_2:.*]] = stablehlo.convert %[[DIM_3_1]] : (tensor<i32>) -> tensor<i64>
  // CHECK: %[[DIM_3:.*]] = stablehlo.reshape %[[DIM_3_2]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[DIM_4:.*]] = stablehlo.constant dense<8> : tensor<1xi64>
  // CHECK: %[[OUTPUT_DIMS:.*]] = stablehlo.concatenate
  // CHECK-SAME: %[[DIM_1]], %[[DIM_2]], %[[DIM_3]], %[[DIM_4]]

  // CHECK: %[[RHS_ZP_BCAST:.*]] = stablehlo.dynamic_broadcast_in_dim
  // CHECK-SAME: %[[RHS_ZP_CONTRIB]], %[[OUTPUT_DIMS]]
  // CHECK-SAME: dims = [0, 3, 1]
  // CHECK-SAME: (tensor<?x8x3xi32>, tensor<4xi64>) -> tensor<?x3x?x8xi32>

  // Combine dot result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = stablehlo.convert %[[DOT_RES]]
  // CHECK-SAME: (tensor<?x3x?x8xi32>) -> tensor<?x3x?x8xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = stablehlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<?x3x?x8xf32>) -> tensor<?x3x?x8xi32>

  // CHECK: %[[ZP_TOTAL_1:.*]] = stablehlo.convert %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<?x3x?x8xi32>) -> tensor<?x3x?x8xf32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_1:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_3:.*]] = stablehlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<?x3x?x8xf32>) -> tensor<?x3x?x8xi32>

  // CHECK: %[[RES_ZP:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_3]]
  // CHECK-SAME: (tensor<i32>, tensor<?x3x?x8xi32>) -> tensor<?x3x?x8xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_4]]

  %0 = "stablehlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 2],
      rhs_batching_dimensions = [1, 4],
      lhs_contracting_dimensions = [4, 3],
      rhs_contracting_dimensions = [0, 2]
    >} : (
      tensor<?x?x3x?x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<6x?x?x8x3x!quant.uniform<i8:f32, 1.000000e+00:0>>
    ) -> tensor<?x3x?x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
  return %0 : tensor<?x3x?x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @dot_general_per_channel
func.func @dot_general_per_channel(
    %arg0: tensor<?x2x!quant.uniform<i8:f32, 2.0:3>>,
    %arg1: tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.0,4.0}>>
  ) -> tensor<?x2x!quant.uniform<i32:f32:1, {6.0,8.0}>> {
  // CHECK: %[[DOT_RES:.*]] = stablehlo.dot_general
  // CHECK-SAME: contracting_dims = [1] x [0]

  // Zero point offset contribution from RHS tensor * LHS ZP.

  // CHECK: %[[RHS_I32:.*]] = stablehlo.convert %arg1 : (tensor<2x2xi8>)
  // CHECK-SAME: -> tensor<2x2xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = stablehlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies stablehlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<2x2xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<2xi32>
  // CHECK: %[[RHS_ZP:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>

  // Calculate output dynamic dims.
  // CHECK: %[[DIM_1_1:.*]] = stablehlo.get_dimension_size %[[DOT_RES]]
  // CHECK-SAME: dim = 0
  // CHECK: %[[DIM_1_2:.*]] = stablehlo.convert %[[DIM_1_1]] : (tensor<i32>) -> tensor<i64>
  // CHECK: %[[DIM_1:.*]] = stablehlo.reshape %[[DIM_1_2]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[DIM_2:.*]] = stablehlo.constant dense<2> : tensor<1xi64>
  // CHECK: %[[OUTPUT_DIMS:.*]] = stablehlo.concatenate
  // CHECK-SAME: %[[DIM_1]], %[[DIM_2]]

  // CHECK: %[[RHS_ZP_BCAST:.*]] = stablehlo.dynamic_broadcast_in_dim
  // CHECK-SAME: %[[RHS_ZP_CONTRIB]], %[[OUTPUT_DIMS]]
  // CHECK-SAME: dims = [1]
  // CHECK-SAME: (tensor<2xi32>, tensor<2xi64>) -> tensor<?x2xi32>
  // CHECK: %[[ZPS_INIT:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_subtract %[[ZPS_INIT]], %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<i32>, tensor<?x2xi32>) -> tensor<?x2xi32>
  // CHECK: chlo.broadcast_add %[[DOT_RES]], %[[ZP_TOTAL_2]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]>} : (
    tensor<?x2x!quant.uniform<i8:f32, 2.0:3>>,
    tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.0,4.0}>>
  ) -> tensor<?x2x!quant.uniform<i32:f32:1, {6.0,8.0}>>
  return %0 : tensor<?x2x!quant.uniform<i32:f32:1, {6.0,8.0}>>
}

// -----

// CHECK-LABEL: func @conv2d_dynamic
func.func @conv2d_dynamic(
    %arg0: tensor<?x?x?x?x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<?x?x?x?x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<?x?x?x?x!quant.uniform<i32:f32, 1.000000e+00:5>> {
  // CHECK-NOT: stablehlo.pad

  // CHECK: %[[CONV:.*]] = stablehlo.convolution
  // CHECK-SAME: (%[[LHS:.*]], %[[RHS:.{1,4}]])
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME: window = {stride = [1, 2], pad = {{\[}}[0, 0], [0, 0]],
  // CHECK-SAME: lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: (tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>) -> tensor<?x?x?x?xi32>

  // Zero point offset contribution from LHS ZP * RHS.

  // CHECK: %[[RHS_I32:.*]] = stablehlo.convert %[[RHS]]
  // CHECK-SAME: (tensor<?x?x?x?xi8>) -> tensor<?x?x?x?xi32>
  // CHECK: %[[RHS_REDUCE:.*]] = stablehlo.reduce(%[[RHS_I32]]
  // CHECK-SAME: applies stablehlo.add across dimensions = [0, 1, 2]
  // CHECK-SAME: (tensor<?x?x?x?xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<?xi32>
  // CHECK: %[[LHS_ZP:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply %[[RHS_REDUCE]], %[[LHS_ZP]]
  // CHECK-SAME: (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = stablehlo.dynamic_broadcast_in_dim
  // CHECK-SAME: %[[RHS_ZP_CONTRIB]]
  // CHECK-SAME: dims = [3]
  // CHECK-SAME: (tensor<?xi32>, tensor<4xi64>) -> tensor<?x?x?x?xi32>

  // Combine conv result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<6.000000e+00> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = stablehlo.convert %[[CONV]]
  // CHECK-SAME: (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = stablehlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi32>

  // CHECK: %[[ZP_TOTAL_1:.*]] = stablehlo.convert %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_1:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_3:.*]] = stablehlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi32>

  // CHECK: %[[RES_ZP:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_3]]
  // CHECK-SAME: (tensor<i32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_4]]
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 2], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [2, 2]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<?x?x?x?x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<?x?x?x?x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<?x?x?x?x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return %0 : tensor<?x?x?x?x!quant.uniform<i32:f32, 1.000000e+00:5>>
}

// -----

// CHECK-LABEL: func @conv2d_static
func.func @conv2d_static(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>> {
  // CHECK-NOT: stablehlo.pad

  // CHECK: %[[CONV:.*]] = stablehlo.convolution
  // CHECK-SAME: (%[[LHS:.*]], %[[RHS:.{1,4}]])
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME: window = {stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]],
  // CHECK-SAME: lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: (tensor<128x28x28x1xi8>, tensor<3x3x1x128xi8>) -> tensor<128x26x26x128xi32>

  // Zero point offset contribution from LHS ZP * RHS.

  // CHECK: %[[RHS_I32:.*]] = stablehlo.convert %[[RHS]]
  // CHECK-SAME: (tensor<3x3x1x128xi8>) -> tensor<3x3x1x128xi32>
  // CHECK: %[[RHS_REDUCE:.*]] = stablehlo.reduce(%[[RHS_I32]]
  // CHECK-SAME: applies stablehlo.add across dimensions = [0, 1, 2]
  // CHECK-SAME: (tensor<3x3x1x128xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<128xi32>
  // CHECK: %[[LHS_ZP:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply %[[RHS_REDUCE]], %[[LHS_ZP]]
  // CHECK-SAME: (tensor<128xi32>, tensor<i32>) -> tensor<128xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = stablehlo.broadcast_in_dim
  // CHECK-SAME: %[[RHS_ZP_CONTRIB]]
  // CHECK-SAME: dims = [3]
  // CHECK-SAME: (tensor<128xi32>) -> tensor<128x26x26x128xi32>

  // Combine conv result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<6.000000e+00> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = stablehlo.convert %[[CONV]]
  // CHECK-SAME: (tensor<128x26x26x128xi32>) -> tensor<128x26x26x128xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = stablehlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<128x26x26x128xf32>) -> tensor<128x26x26x128xi32>

  // CHECK: %[[ZP_TOTAL_1:.*]] = stablehlo.convert %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<128x26x26x128xi32>) -> tensor<128x26x26x128xf32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_1:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_3:.*]] = stablehlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<128x26x26x128xf32>) -> tensor<128x26x26x128xi32>

  // CHECK: %[[RES_ZP:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_3]]
  // CHECK-SAME: (tensor<i32>, tensor<128x26x26x128xi32>) -> tensor<128x26x26x128xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_4]]
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return %0 : tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
}

// -----

// CHECK-LABEL: func @conv2d_default_attr
func.func @conv2d_default_attr(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>> {
  // CHECK: stablehlo.convolution
  // CHECK-NOT: quant.uniform
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return %0 : tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
}

// -----

// CHECK-LABEL: func @conv2d_static_padding
func.func @conv2d_static_padding(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<128x29x33x128x!quant.uniform<i32:f32, 1.000000e+00:5>> {
  // Explicitly pad LHS with ZP.

  // CHECK: %[[LHS_ZP_i8:.*]] = stablehlo.constant dense<4> : tensor<i8>
  // CHECK: %[[LHS_PAD:.*]] = stablehlo.pad %[[LHS:.*]], %[[LHS_ZP_i8]]
  // CHECK-SAME: low = [0, 1, 3, 0]
  // CHECK-SAME: high = [0, 2, 4, 0]
  // CHECK-SAME: interior = [0, 0, 0, 0]
  // CHECK-SAME: (tensor<128x28x28x1xi8>, tensor<i8>) -> tensor<128x31x35x1xi8>

  // Convolution with padding removed.

  // CHECK: %[[CONV:.*]] = stablehlo.convolution
  // CHECK-SAME: (%[[LHS_PAD]], %[[RHS:.{1,4}]])
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME: window = {stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]],
  // CHECK-SAME: lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: (tensor<128x31x35x1xi8>, tensor<3x3x1x128xi8>) -> tensor<128x29x33x128xi32>

  // Zero point offset contribution from LHS ZP * RHS.

  // CHECK: %[[RHS_I32:.*]] = stablehlo.convert %[[RHS]]
  // CHECK-SAME: (tensor<3x3x1x128xi8>) -> tensor<3x3x1x128xi32>
  // CHECK: %[[RHS_REDUCE:.*]] = stablehlo.reduce(%[[RHS_I32]]
  // CHECK-SAME: applies stablehlo.add across dimensions = [0, 1, 2]
  // CHECK-SAME: (tensor<3x3x1x128xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<128xi32>
  // CHECK: %[[LHS_ZP:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply %[[RHS_REDUCE]], %[[LHS_ZP]]
  // CHECK-SAME: (tensor<128xi32>, tensor<i32>) -> tensor<128xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = stablehlo.broadcast_in_dim
  // CHECK-SAME: %[[RHS_ZP_CONTRIB]]
  // CHECK-SAME: dims = [3]
  // CHECK-SAME: (tensor<128xi32>) -> tensor<128x29x33x128xi32>

  // Combine conv result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = stablehlo.constant dense<6.000000e+00> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = stablehlo.convert %[[CONV]]
  // CHECK-SAME: (tensor<128x29x33x128xi32>) -> tensor<128x29x33x128xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = stablehlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<128x29x33x128xf32>) -> tensor<128x29x33x128xi32>

  // CHECK: %[[ZP_TOTAL_1:.*]] = stablehlo.convert %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<128x29x33x128xi32>) -> tensor<128x29x33x128xf32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_1:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_3:.*]] = stablehlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<128x29x33x128xf32>) -> tensor<128x29x33x128xi32>

  // CHECK: %[[RES_ZP:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_3]]
  // CHECK-SAME: (tensor<i32>, tensor<128x29x33x128xi32>) -> tensor<128x29x33x128xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_4]]
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[1, 2], [3, 4]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x29x33x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return %0 : tensor<128x29x33x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
}

// -----

// CHECK-LABEL: func @conv2d_per_channel
func.func @conv2d_per_channel(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>> {
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
  // CHECK-SAME: window = {stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]],
  // CHECK-SAME: lhs_dilate = [1, 1], rhs_dilate = [1, 1]
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: (tensor<128x28x28x1xi8>, tensor<3x3x1x2xi8>) -> tensor<128x26x26x2xi32>

  // CHECK: %[[RHS:.*]] = stablehlo.convert %arg1 : (tensor<3x3x1x2xi8>) -> tensor<3x3x1x2xi32>
  // CHECK: %[[REDUCE:.*]] = stablehlo.reduce(%[[RHS]]
  // CHECK-SAME: applies stablehlo.add across dimensions = [0, 1, 2]
  // CHECK: %[[LHS_ZP:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK: %[[ZP_OFFSET:.*]] = chlo.broadcast_multiply %[[REDUCE]], %[[LHS_ZP]]
  // CHECK: %[[ZP_OFFSET_BCAST:.*]] = stablehlo.broadcast_in_dim %[[ZP_OFFSET]]
  // CHECK: %[[RES_ZP:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[ZP_OFFSET_TOTAL:.*]] = chlo.broadcast_subtract %[[RES_ZP:.*]], %[[ZP_OFFSET_BCAST]]
  // CHECK: chlo.broadcast_add %[[CONV]], %[[ZP_OFFSET_TOTAL]]
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (
      tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
}

// -----

// CHECK-LABEL: func @conv3d_static
func.func @conv3d_static(
    %arg0: tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<128x26x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>{
  // CHECK-NOT: stablehlo.pad

  // CHECK: stablehlo.convolution
  // CHECK-SAME: dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
  // CHECK-SAME: window = {stride = [1, 1, 1], pad = {{\[}}[0, 0], [0, 0], [0, 0]],
  // CHECK-SAME: lhs_dilate = [1, 1, 1], rhs_dilate = [1, 1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: (tensor<128x28x28x28x1xi8>, tensor<3x3x3x1x128xi8>) -> tensor<128x26x26x26x128xi32>

  // CHECK: stablehlo.reduce
  // CHECK-SAME: applies stablehlo.add across dimensions = [0, 1, 2, 3]
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f],
    window = {
      stride = [1, 1, 1], pad = [[0, 0], [0, 0], [0, 0]],
      lhs_dilate = [1, 1, 1],
      rhs_dilate = [1, 1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x26x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return %0 : tensor<128x26x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
}

// -----

func.func @conv3d_rhs_zp_not_zero(
    %arg0: tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:-2>>) {
  // CHECK: stablehlo.convolution{{.*}} : (tensor<128x28x28x28x1xf32>, tensor<3x3x3x1x128xf32>) -> tensor<128x26x26x26x128xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f],
    window = {
      stride = [1, 1, 1], pad = [[0, 0], [0, 0], [0, 0]],
      lhs_dilate = [1, 1, 1],
      rhs_dilate = [1, 1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:-2>>)
    -> tensor<128x26x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

func.func @conv3d_rhs_invalid_dilate(
    %arg0: tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>) {
  // CHECK: stablehlo.convolution{{.*}} : (tensor<128x28x28x28x1xf32>, tensor<3x3x3x1x128xf32>) -> tensor<128x53x53x53x128xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f],
    window = {
      stride = [1, 1, 1], pad = [[0, 0], [0, 0], [0, 0]],
      lhs_dilate = [2, 2, 2],
      rhs_dilate = [1, 1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x53x53x53x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

func.func @conv3d_non_nhwc(
    %arg0: tensor<128x1x28x28x28x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>) {
  // CHECK: stablehlo.convolution{{.*}} : (tensor<128x1x28x28x28xf32>, tensor<3x3x3x1x128xf32>) -> tensor<128x128x26x26x26xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1, 2]x[0, 1, 2, i, o]->[b, f, 0, 1, 2],
    window = {
      stride = [1, 1, 1], pad = [[0, 0], [0, 0], [0, 0]],
      lhs_dilate = [1, 1, 1],
      rhs_dilate = [1, 1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x1x28x28x28x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x128x26x26x26x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

func.func @conv2d_non_nhwc(
    %arg0: tensor<128x1x28x28x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>) {
  // CHECK: stablehlo.convolution{{.*}} : (tensor<128x1x28x28xf32>, tensor<3x3x1x128xf32>) -> tensor<128x128x26x26xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x1x28x28x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x128x26x26x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

func.func @conv2d_per_channel_rhs_zp_not_zero(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:10}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>> {
  // CHECK: stablehlo.convolution{{.*}} : (tensor<128x28x28x1xf32>, tensor<3x3x1x2xf32>) -> tensor<128x26x26x2xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (
      tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:10}>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
}

// -----

func.func @conv2d_per_channel_res_zp_not_zero(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:3}>> {
  // CHECK: stablehlo.convolution{{.*}} : (tensor<128x28x28x1xf32>, tensor<3x3x1x2xf32>) -> tensor<128x26x26x2xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (
      tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:3}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:3}>>
}

// -----

func.func @conv2d_per_channel_rhs_only(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32, 4.000000e+00:0>> {
  // CHECK: stablehlo.convolution{{.*}} : (tensor<128x28x28x1xf32>, tensor<3x3x1x2xf32>) -> tensor<128x26x26x2xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (
      tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32, 4.000000e+00:0>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32, 4.000000e+00:0>>
}

// -----

func.func @conv2d_per_channel_rhs_result_scale_ratio_different(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.200000e+00:0}>> {
  // CHECK: stablehlo.convolution{{.*}} : (tensor<128x28x28x1xf32>, tensor<3x3x1x2xf32>) -> tensor<128x26x26x2xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (
      tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.200000e+00:0}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.200000e+00:0}>>
}

// -----

// CHECK-LABEL: func @dot_hybrid
func.func @dot_hybrid(
    %arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32> {
  // CHECK: %[[VAL1:.*]] = stablehlo.optimization_barrier %[[VAL0:.*]] : tensor<?x?xi8>
  // CHECK-DAG: %[[VAL5:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[VAL3:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  // CHECK: %[[VAL2:.*]] = stablehlo.convert %[[VAL1:.*]] : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_subtract %[[VAL2]], %[[VAL3:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_multiply %[[VAL4]], %[[VAL5:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL7:.*]] = stablehlo.dot %arg0, %[[VAL6]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "stablehlo.dot" (%arg0, %arg1): (
      tensor<?x?xf32>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1: tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @dot_general_hybrid_per_channel
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x2xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<2x2xi8>
func.func @dot_general_hybrid_per_channel(
    %arg0: tensor<3x2xf32>,
    %arg1: tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.000000e+00, 4.000000e+00}>>
  ) -> tensor<3x2xf32> {
  // CHECK-DAG: %[[BARRIER:.*]] = stablehlo.optimization_barrier %[[ARG1]] : tensor<2x2xi8>
  // CHECK-DAG: %[[SCALES:.*]] = stablehlo.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>
  // CHECK-DAG: %[[CONVERT:.*]] = stablehlo.convert %[[BARRIER]] : (tensor<2x2xi8>) -> tensor<2x2xf32>
  // CHECK-NOT: chlo.broadcast_subtract
  // CHECK: %[[MUL:.*]] = chlo.broadcast_multiply %[[CONVERT]], %[[SCALES]] {broadcast_dimensions = array<i64: 1>} : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG0]], %[[MUL]]
  // CHECK-SAME: (tensor<3x2xf32>, tensor<2x2xf32>) -> tensor<3x2xf32>
  // CHECK: return %[[DOT]]

  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]>} : (
    tensor<3x2xf32>,
    tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.000000e+00, 4.000000e+00}>>
  ) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: func @dot_general_hybrid_per_channel_asymmetric
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x2xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<2x2xi8>
func.func @dot_general_hybrid_per_channel_asymmetric(
    %arg0: tensor<3x2xf32>,
    %arg1: tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.000000e+00:0, 4.000000e+00:0}>>
  ) -> tensor<3x2xf32> {
  // CHECK-DAG: %[[BARRIER:.*]] = stablehlo.optimization_barrier %[[ARG1]] : tensor<2x2xi8>
  // CHECK-DAG: %[[SCALES:.*]] = stablehlo.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>
  // CHECK-DAG: %[[ZPS:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
  // CHECK-DAG: %[[CONVERT:.*]] = stablehlo.convert %[[BARRIER]] : (tensor<2x2xi8>) -> tensor<2x2xf32>
  // CHECK: %[[MUL:.*]] = chlo.broadcast_multiply %[[CONVERT]], %[[SCALES]] {broadcast_dimensions = array<i64: 1>} : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG0]], %[[MUL]]
  // CHECK-SAME: (tensor<3x2xf32>, tensor<2x2xf32>) -> tensor<3x2xf32>
  // CHECK: return %[[DOT]]

  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]>} : (
    tensor<3x2xf32>,
    tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.000000e+00:0, 4.000000e+00:0}>>
  ) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// -----

func.func @dot_hybrid_result_type_not_float(
    %arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) {
  // CHECK: stablehlo.dot {{.*}} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "stablehlo.dot" (%arg0, %arg1): (
      tensor<?x?xf32>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return
}

// -----

func.func @dot_hybrid_lhs_type_not_float(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>,
    %arg1: tensor<?x?xf32>) {
  // CHECK: stablehlo.dot {{.*}} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "stablehlo.dot" (%arg0, %arg1): (
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<?x?xf32>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return
}

// -----

// CHECK-LABEL: func @conv2d_static_hybrid
func.func @conv2d_static_hybrid(
    %arg0: tensor<128x28x28x1xf32>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:1>>
  ) -> tensor<128x26x26x128xf32> {
  // CHECK-DAG: %[[BARRIER:.*]] = stablehlo.optimization_barrier %arg1 : tensor<3x3x1x128xi8>
  // CHECK-DAG: %[[ZP:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[SCALE:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  // CHECK: %[[RHS:.*]] = stablehlo.convert %[[BARRIER]] : (tensor<3x3x1x128xi8>) -> tensor<3x3x1x128xf32>
  // CHECK: %[[SUB:.*]] = chlo.broadcast_subtract %[[RHS]], %[[ZP]]
  // CHECK: %[[MUL:.*]] = chlo.broadcast_multiply %[[SUB]], %[[SCALE]]
  // CHECK: stablehlo.convolution(%arg0, %[[MUL]])
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME: stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]]
  // CHECK-SAME: lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: : (tensor<128x28x28x1xf32>, tensor<3x3x1x128xf32>) -> tensor<128x26x26x128xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x1xf32>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:1>>)
    -> tensor<128x26x26x128xf32>
  return %0 : tensor<128x26x26x128xf32>
}

// -----

// CHECK-LABEL: func @conv2d_hybrid_per_channel
// CHECK-SAME: %[[ARG0:.*]]: tensor<128x28x28x1xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<3x3x1x2xi8>
func.func @conv2d_hybrid_per_channel(
    %arg0: tensor<128x28x28x1xf32>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>
  ) -> tensor<128x26x26x2xf32> {
  // CHECK-DAG: %[[BARRIER:.*]] = stablehlo.optimization_barrier %[[ARG1]] : tensor<3x3x1x2xi8>
  // CHECK-DAG: %[[SCALES:.*]] = stablehlo.constant dense<[2.000000e+00, 1.000000e+00]> : tensor<2xf32>
  // CHECK-DAG: %[[CONVERT:.*]] = stablehlo.convert %[[BARRIER]] : (tensor<3x3x1x2xi8>) -> tensor<3x3x1x2xf32>
  // CHECK-NOT: chlo.broadcast_subtract
  // CHECK: %[[MUL:.*]] = chlo.broadcast_multiply %[[CONVERT]], %[[SCALES]] {broadcast_dimensions = array<i64: 3>} : (tensor<3x3x1x2xf32>, tensor<2xf32>) -> tensor<3x3x1x2xf32>
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[ARG0]], %[[MUL]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x28x28x1xf32>, tensor<3x3x1x2xf32>) -> tensor<128x26x26x2xf32>
  // CHECK: return %[[CONV]]

  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (
      tensor<128x28x28x1xf32>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>)
    -> tensor<128x26x26x2xf32>
  return %0 : tensor<128x26x26x2xf32>
}

// -----

// CHECK-LABEL: func @conv2d_hybrid_per_channel_asymmetric
// CHECK-SAME: %[[ARG0:.*]]: tensor<128x28x28x1xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<3x3x1x2xi8>
func.func @conv2d_hybrid_per_channel_asymmetric(
    %arg0: tensor<128x28x28x1xf32>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:10, 1.000000e+00:20}>>
  ) -> tensor<128x26x26x2xf32> {
  // CHECK-DAG: %[[BARRIER:.*]] = stablehlo.optimization_barrier %[[ARG1]] : tensor<3x3x1x2xi8>
  // CHECK-DAG: %[[SCALES:.*]] = stablehlo.constant dense<[2.000000e+00, 1.000000e+00]> : tensor<2xf32>
  // CHECK-DAG: %[[ZPS:.*]] = stablehlo.constant dense<[1.000000e+01, 2.000000e+01]> : tensor<2xf32>
  // CHECK-DAG: %[[CONVERT:.*]] = stablehlo.convert %[[BARRIER]] : (tensor<3x3x1x2xi8>) -> tensor<3x3x1x2xf32>
  // CHECK: %[[SUB:.*]] = chlo.broadcast_subtract %[[CONVERT]], %[[ZPS]] {broadcast_dimensions = array<i64: 3>} : (tensor<3x3x1x2xf32>, tensor<2xf32>) -> tensor<3x3x1x2xf32>
  // CHECK: %[[MUL:.*]] = chlo.broadcast_multiply %[[SUB]], %[[SCALES]] {broadcast_dimensions = array<i64: 3>} : (tensor<3x3x1x2xf32>, tensor<2xf32>) -> tensor<3x3x1x2xf32>
  // CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[ARG0]], %[[MUL]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x28x28x1xf32>, tensor<3x3x1x2xf32>) -> tensor<128x26x26x2xf32>
  // CHECK: return %[[CONV]]

  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (
      tensor<128x28x28x1xf32>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:10, 1.000000e+00:20}>>)
    -> tensor<128x26x26x2xf32>
  return %0 : tensor<128x26x26x2xf32>
}

// -----

func.func @conv2d_hybrid_result_not_float(
    %arg0: tensor<128x28x28x1xf32>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>) {
  // expected-error@+1 {{rhs should be quantized for quantized operations and is_quantized(lhs)=is_quantized(result) should hold}}
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x1xf32>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

func.func @dot_general_non_hybrid_result_not_float(
  %arg0: tensor<2x5x6x!quant.uniform<i8:f32, 1.000000e+00:0>>,
  %arg1: tensor<6x8x2x!quant.uniform<i8:f32:2, {1.000000e+00:0, 1.000000e+00:0}>>) {
  // CHECK: stablehlo.dot_general {{.*}} : (tensor<2x5x6xf32>, tensor<6x8x2xf32>) -> tensor<2x5x8xf32>
  %0 = "stablehlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6x!quant.uniform<i8:f32, 1.000000e+00:0>>,
      tensor<6x8x2x!quant.uniform<i8:f32:2, {1.000000e+00:0, 1.000000e+00:0}>>
    ) -> tensor<2x5x8x!quant.uniform<i8:f32, 4.000000e+00:7>>
  return
}

// -----

// CHECK-LABEL: func @mhlo_constant_uniform_quantized
func.func @mhlo_constant_uniform_quantized() -> tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>> {
  // CHECK: stablehlo.constant dense<9> : tensor<1xi8>
  %0 = stablehlo.constant() {value = dense<9> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return %0 : tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @mhlo_constant_uniform_quantized_per_channel
func.func @mhlo_constant_uniform_quantized_per_channel() -> () {
  // CHECK: stablehlo.constant dense<[9, 4]> : tensor<2xi8>
  %0 = stablehlo.constant() {value = dense<[9, 4]> : tensor<2xi8>} : ()
      -> tensor<2x!quant.uniform<i8:f32:0, {1.000000e+00:3, 2.000000e+00:-2}>>
  return
}


// -----

// CHECK-LABEL: func @mhlo_constant_int
func.func @mhlo_constant_int() -> tensor<i32> {
  // CHECK: stablehlo.constant dense<-128> : tensor<i32>
  %0 = stablehlo.constant() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @broadcast
func.func @broadcast(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<2x3x1x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: stablehlo.broadcast_in_dim
  // CHECK-SAME: dims = [2, 0]
  // CHECK-SAME: (tensor<1x2xi8>) -> tensor<2x3x1xi8>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64:2, 0>
    } : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>) -> tensor<2x3x1x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<2x3x1x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @broadcast_per_channel
func.func @broadcast_per_channel(
    %arg0: tensor<2x!quant.uniform<i32:f32:0, {4.000000e+00:0, 2.000000e+00:0}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>  {
  // CHECK: stablehlo.broadcast_in_dim
  // CHECK-SAME: dims = [3]
  // CHECK-SAME: (tensor<2xi32>) -> tensor<128x26x26x2xi32>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64:3>} : (
      tensor<2x!quant.uniform<i32:f32:0, {4.000000e+00:0, 2.000000e+00:0}>>
    ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast
func.func @dynamic_broadcast(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<3xi32>
  ) -> tensor<?x1x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK-SAME: dims = [1, 2]
  // CHECK-SAME: (tensor<1x2xi8>, tensor<3xi32>) -> tensor<?x1x2xi8>
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {
      broadcast_dimensions = array<i64:1, 2>
    } : (
      tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<3xi32>
    ) -> tensor<?x1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<?x1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @max_per_tensor_same_quant_parameters
func.func @max_per_tensor_same_quant_parameters(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: stablehlo.maximum
  // CHECK-SAME: tensor<1x2xi8>
  %0 = "stablehlo.maximum"(%arg0, %arg0) : (
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @max_per_channel_same_quant_parameters
func.func @max_per_channel_same_quant_parameters(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: stablehlo.maximum
  // CHECK-SAME: tensor<1x2xi8>
  %0 = "stablehlo.maximum"(%arg0, %arg0) : (
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

// CHECK-LABEL: func.func @max_per_tensor_diff_quant_parameters
func.func @max_per_tensor_diff_quant_parameters(%arg0: tensor<!quant.uniform<i8:f32,1.0:0>>, %arg1: tensor<!quant.uniform<i8:f32,2.0:1>>) ->  tensor<!quant.uniform<i8:f32,3.0:2>> {
  // CHECK: stablehlo.maximum {{.*}} : tensor<f32>
  %0 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<!quant.uniform<i8:f32,1.0:0>>, tensor<!quant.uniform<i8:f32,2.0:1>>) -> tensor<!quant.uniform<i8:f32,3.0:2>>
  func.return %0 : tensor<!quant.uniform<i8:f32,3.0:2>>
}

// -----

// CHECK-LABEL: func.func @min_per_tensor_diff_quant_parameters
func.func @min_per_tensor_diff_quant_parameters(%arg0: tensor<!quant.uniform<i8:f32,1.0:0>>, %arg1: tensor<!quant.uniform<i8:f32,2.0:1>>) ->  tensor<!quant.uniform<i8:f32,3.0:2>> {
  // CHECK: stablehlo.minimum {{.*}} : tensor<f32>
  %0 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<!quant.uniform<i8:f32,1.0:0>>, tensor<!quant.uniform<i8:f32,2.0:1>>) -> tensor<!quant.uniform<i8:f32,3.0:2>>
  func.return %0 : tensor<!quant.uniform<i8:f32,3.0:2>>
}

// -----

// CHECK-LABEL: func @min_per_tensor_same_quant_parameters
func.func @min_per_tensor_same_quant_parameters(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: stablehlo.minimum
  // CHECK-SAME: tensor<1x2xi8>
  %0 = "stablehlo.minimum"(%arg0, %arg0) : (
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @min_per_channel_same_quant_parameters
func.func @min_per_channel_same_quant_parameters(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: stablehlo.minimum
  // CHECK-SAME: tensor<1x2xi8>
  %0 = "stablehlo.minimum"(%arg0, %arg0) : (
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @function(%arg0: tensor<1x2xi8>) -> tensor<1x2xi8>
func.func @function(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: return %arg0 : tensor<1x2xi8>
  return %arg0 : tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @concatenate
func.func @concatenate(
    %arg0: tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>,
    %arg1: tensor<1x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  ) -> tensor<4x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>> {
  // CHECK: stablehlo.concatenate
  // CHECK-SAME: (tensor<3x2xi8>, tensor<1x2xi8>) -> tensor<4x2xi8>
  %0 = "stablehlo.concatenate"(%arg0, %arg1) <{dimension = 0 : i64}> : (
    tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>,
    tensor<1x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  ) -> tensor<4x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  return %0 : tensor<4x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
}

// -----

// CHECK-LABEL: func @pad
func.func @pad(
    %arg0: tensor<2x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>,
    %arg1: tensor<!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  ) -> tensor<5x9x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>> {
  // CHECK: stablehlo.pad
  // CHECK-SAME: (tensor<2x3xi8>, tensor<i8>) -> tensor<5x9xi8>
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1>,
    edge_padding_high = array<i64: 2, 1>,
    interior_padding = array<i64: 1, 2>
  }: (
    tensor<2x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>,
    tensor<!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  ) -> tensor<5x9x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  return %0 : tensor<5x9x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
}

// -----

// CHECK-LABEL: func @reshape
func.func @reshape(
    %arg0: tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: stablehlo.reshape
  // CHECK-SAME: (tensor<1x3xi8>) -> tensor<3x1xi8>
  %0 = "stablehlo.reshape"(%arg0) : (
    tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @dynamic_reshape
func.func @dynamic_reshape(
    %arg0: tensor<?x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    %arg1: tensor<2xi32>
  ) -> tensor<?x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: stablehlo.dynamic_reshape
  // CHECK-SAME: (tensor<?x3xi8>, tensor<2xi32>) -> tensor<?x1xi8>
  %0 = "stablehlo.dynamic_reshape"(%arg0, %arg1) : (
    tensor<?x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>, tensor<2xi32>
  ) -> tensor<?x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<?x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @select
func.func @select(
    %arg0: tensor<1x3xi1>,
    %arg1: tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    %arg2: tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: stablehlo.select
  // CHECK-SAME: tensor<1x3xi8>
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (
    tensor<1x3xi1>,
    tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @transpose
func.func @transpose(
    %arg0: tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: stablehlo.transpose
  // CHECK-SAME: (tensor<3x1xi8>) -> tensor<1x3xi8>
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0>}  : (
    tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @gather
func.func @gather(
    %arg0: tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    %arg1:  tensor<2x3x2xi64>
  ) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: stablehlo.gather
  // CHECK-SAME: (tensor<3x4x2xi8>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi8>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2>,
    indices_are_sorted = false
  } : (
    tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    tensor<2x3x2xi64>
  ) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @slice
func.func @slice(
    %arg0: tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: stablehlo.slice
  // CHECK-SAME: (tensor<3x4xi8>) -> tensor<2x2xi8>
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 2>,
    limit_indices = array<i64: 3, 4>,
    strides = array<i64:1, 1>
  } : (
    tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @dynamic_slice
func.func @dynamic_slice(
    %arg0: tensor<?x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    %arg1: tensor<i32>,
    %arg2: tensor<i32>
  ) -> tensor<1x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: stablehlo.dynamic_slice
  // CHECK-SAME: (tensor<?x4xi8>, tensor<i32>, tensor<i32>) -> tensor<1x1xi8>
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {
    slice_sizes = array<i64:1, 1>
  } : (
    tensor<?x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>, tensor<i32>,
    tensor<i32>
  ) -> tensor<1x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<1x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @get_dimension_size
func.func @get_dimension_size(
    %arg0: tensor<?x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<i32> {
  // CHECK: stablehlo.get_dimension_size
  // CHECK-SAME: (tensor<?x4xi8>) -> tensor<i32>
  %0 = "stablehlo.get_dimension_size"(%arg0) <{dimension = 0 : i64}> : (
      tensor<?x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: reduce_window
func.func @reduce_window(
    %arg0: tensor<2x3x10x3x!quant.uniform<i8:f32, 3.000000e-01:-49>>,
    %arg1: tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>
  ) -> tensor<2x3x10x3x!quant.uniform<i8:f32, 3.000000e-01:-49>> {
  // CHECK: stablehlo.reduce_window
  // CHECK: %[[ARG2:.*]]: tensor<i8>, %[[ARG3:.*]]: tensor<i8>
  // CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ARG2]], %[[ARG3]] : tensor<i8>
  // CHECK: stablehlo.return %[[MAX]] : tensor<i8>
  // CHECK: (tensor<2x3x10x3xi8>, tensor<i8>) -> tensor<2x3x10x3xi8>
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>, %arg3: tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>):
    %1 = stablehlo.maximum %arg2, %arg3 : tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>
    stablehlo.return %1 : tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>
  }) {
    padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dimensions = array<i64: 1, 3, 3, 1>
  } : (tensor<2x3x10x3x!quant.uniform<i8:f32, 3.000000e-01:-49>>, tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>) -> tensor<2x3x10x3x!quant.uniform<i8:f32, 3.000000e-01:-49>>
  return %0 : tensor<2x3x10x3x!quant.uniform<i8:f32, 3.000000e-01:-49>>
}

// -----

// CHECK-LABEL: func.func @miscellaneous_quantized_ops
func.func @miscellaneous_quantized_ops(
  %arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>,
  %arg1: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>,
  %arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>,
  %arg3: tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>,
  %shape: tensor<3xi64>, %token0: !stablehlo.token) {
  %abs = "stablehlo.abs"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %atan2 = "stablehlo.atan2"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %bitcast_convert = "stablehlo.bitcast_convert"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %broadcast_in_dim = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %cbrt = "stablehlo.cbrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %ceil = "stablehlo.ceil"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %cholesky = "stablehlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %compare = "stablehlo.compare"(%arg0, %arg1) { comparison_direction = #stablehlo<comparison_direction LT>, compare_type = #stablehlo<comparison_type FLOAT> } : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xi1>
  %concatenate = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %cosine = "stablehlo.cosine"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %divide = "stablehlo.divide"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %dynamic_broadcast_in_dim = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<3xi64>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %exponential = "stablehlo.exponential"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %exponential_minus_one = "stablehlo.exponential_minus_one"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %floor = "stablehlo.floor"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %get_dimension_size = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<i32>
  %log = "stablehlo.log"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %log_plus_one = "stablehlo.log_plus_one"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %logistic = "stablehlo.logistic"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %maximum = "stablehlo.maximum"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %minimum = "stablehlo.minimum"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %multiply = "stablehlo.multiply"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %negate = "stablehlo.negate"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %power = "stablehlo.power"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %remainder = "stablehlo.remainder"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %reshape = "stablehlo.reshape" (%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %rsqrt = "stablehlo.rsqrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %sign = "stablehlo.sign"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %sine = "stablehlo.sine"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %sqrt = "stablehlo.sqrt"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %subtract = "stablehlo.subtract"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %tanh = "stablehlo.tanh"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  %transpose = "stablehlo.transpose"(%arg0) {permutation = array<i64: 0, 2, 1>}: (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8<-128:127>:f32, 1.0:17>>
  %uniform_dequantize = "stablehlo.uniform_dequantize" (%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xf32>
  %uniform_quantize = "stablehlo.uniform_quantize" (%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>

 func.return
}

// CHECK: stablehlo.abs {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.atan2 {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.bitcast_convert {{.*}} : (tensor<1x2x2xi8>) -> tensor<1x2x2xi8>
// CHECK: stablehlo.broadcast_in_dim {{.*}} : (tensor<1x2x2xi8>) -> tensor<1x2x2xi8>
// CHECK: stablehlo.cbrt {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.ceil {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.cholesky {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.compare  LT, {{.*}} : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xi1>
// CHECK: stablehlo.concatenate {{.*}} : (tensor<1x2x2xi8>, tensor<1x2x2xi8>) -> tensor<2x2x2xi8>
// CHECK: stablehlo.cosine {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.divide {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.dynamic_broadcast_in_dim {{.*}} : (tensor<1x2x2xi8>, tensor<3xi64>) -> tensor<1x2x2xi8>
// CHECK: stablehlo.exponential {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.exponential_minus_one {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.floor {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.get_dimension_size {{.*}} : (tensor<1x2x2xi8>) -> tensor<i32>
// CHECK: stablehlo.log {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.log_plus_one {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.logistic {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.maximum {{.*}} : tensor<1x2x2xi8>
// CHECK: stablehlo.minimum {{.*}} : tensor<1x2x2xi8>
// CHECK: stablehlo.multiply {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.negate {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.power {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.remainder {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.reshape {{.*}} : (tensor<1x2x2xi8>) -> tensor<1x2x2xi8>
// CHECK: stablehlo.rsqrt {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.sign {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.sine {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.sqrt {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.subtract {{.*}} : tensor<1x2x2xf32>
// CHECK: stablehlo.tanh {{.*}} : tensor<1x2x2xf32>
    // CHECK: stablehlo.transpose {{.*}} : (tensor<1x2x2xi8>) -> tensor<1x2x2xi8>

// -----

func.func @all_gather(%arg3: tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.all_gather' that was explicitly marked illegal}}
  %0 = "stablehlo.all_gather"(%arg3) { all_gather_dim = 1 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
}

// -----
func.func @all_to_all(%arg3: tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.all_to_all' that was explicitly marked illegal}}
  %0 = "stablehlo.all_to_all"(%arg3) { split_dimension = 1 : i64, concat_dimension = 1 : i64, split_count = 2 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>} : (tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x4x!quant.uniform<i8:f32, 1.0:17>>
}

// -----
func.func @collective_permute(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.collective_permute' that was explicitly marked illegal}}
  %0 = "stablehlo.collective_permute"(%arg0) { source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>, channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

// -----
func.func @custom_call(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.custom_call' that was explicitly marked illegal}}
  %0 = "stablehlo.custom_call" (%arg0) {call_target_name = "foo"} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

// -----
func.func @is_finite(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xi1> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.is_finite' that was explicitly marked illegal}}
  %0 = "stablehlo.is_finite"(%arg0) {} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2xi1>
  func.return %0 : tensor<1x2x2xi1>
}

// -----
func.func @outfeed(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, %token0: !stablehlo.token) -> !stablehlo.token {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.outfeed' that was explicitly marked illegal}}
  %0 = "stablehlo.outfeed"(%arg0, %token0) {outfeed_config = ""} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

// -----
func.func @optimization_barrier(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.optimization_barrier' that was explicitly marked illegal}}
  %0 = "stablehlo.optimization_barrier"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>)
  func.return %0 : tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

// -----
func.func @send(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, %token0: !stablehlo.token) -> !stablehlo.token {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.send' that was explicitly marked illegal}}
  %0 = "stablehlo.send"(%arg0, %token0) {channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>, is_host_transfer = true} : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

// -----

// CHECK-LABEL:  func.func @tan
func.func @tan(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.tan"(%arg0) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: stablehlo.tan {{.*}} : tensor<1x2x2xf32>


// -----
func.func @tuple(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>,
                 %arg1: tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.tuple' that was explicitly marked illegal}}
  %0 = "stablehlo.tuple"(%arg0, %arg1) : (tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tuple<tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x2x2x!quant.uniform<i8:f32, 1.0:17>>>
  func.return
}

// -----

// CHECK-LABEL:  func.func @batch_norm_grad_per_tensor_quantization
func.func @batch_norm_grad_per_tensor_quantization(%input: tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, %scale: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %mean: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %variance: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %grad_output: tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output)
   {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>)
   -> (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>)
  func.return %0#0 : tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: "stablehlo.batch_norm_grad"{{.*}} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)

// -----

// CHECK-LABEL: @batch_norm_inference_per_tensor_quantization
func.func @batch_norm_inference_per_tensor_quantization(%input: tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>, %scale: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %offset: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %mean: tensor<256x!quant.uniform<i8:f32, 1.0:17>>, %variance: tensor<256x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>) {
  %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {
    epsilon = 1.001000e-05 : f32,
    feature_index = 1 : i64
  } : (tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>, tensor<256x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<4x256x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: "stablehlo.batch_norm_inference"{{.*}} : (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256xf32>


// -----

// CHECK-LABEL: @batch_norm_training_per_tensor_quantization
func.func @batch_norm_training_per_tensor_quantization(%input: tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, %scale: tensor<2x!quant.uniform<i8:f32, 1.0:17>>, %offset: tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>> {
  %0:3 = "stablehlo.batch_norm_training" (%input, %scale, %offset) {
    epsilon = 0.001 : f32,
    feature_index = 1 : i64
  } : (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>) ->
      (tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x!quant.uniform<i8:f32, 1.0:17>>)
  func.return %0#0 : tensor<2x2x2x2x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: "stablehlo.batch_norm_training"{{.*}} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)

// -----

// CHECK-LABEL: @dynamic_slice_per_tensor_quantization
func.func @dynamic_slice_per_tensor_quantization(%arg0: tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<i64>, tensor<i64>) -> tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: stablehlo.dynamic_slice {{.*}} : (tensor<3x4xi8>, tensor<i64>, tensor<i64>) -> tensor<1x4xi8>


// -----

func.func @dynamic_update_slice_per_tensor_quantization(%operand: tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, %update: tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.dynamic_update_slice' that was explicitly marked illegal}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<i64>, tensor<i64>) -> tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>
}

// -----

// CHECK-LABEL: @gather_per_tensor_quantization
func.func @gather_per_tensor_quantization(%operand : tensor<?x?x?x?x?x?x?x?x!quant.uniform<i8:f32, 1.0:17>>, %start_indices : tensor<1x5x2xi32>) -> tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32, 1.0:17>> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0, 2, 3, 4, 5],
      collapsed_slice_dims = [0, 1, 3],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8, 1, 7, 1, 6, 1>,
    indices_are_sorted = false
  } : (tensor<?x?x?x?x?x?x?x?x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x5x2xi32>) -> tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32, 1.0:17>>
  func.return %res : tensor<8x?x7x1x6x1x?x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: "stablehlo.gather"{{.*}} :  (tensor<?x?x?x?x?x?x?x?xi8>, tensor<1x5x2xi32>) -> tensor<8x?x7x1x6x1x?xi8>


// -----

func.func @map_per_tensor_quantization(%arg0: tensor<4x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.map' that was explicitly marked illegal}}
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
    "stablehlo.return"(%arg2) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<4x!quant.uniform<i8:f32, 1.0:17>>, tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<4x!quant.uniform<i8:f32, 1.0:17>>
}

// -----

// CHECK-LABEL: @pad_per_tensor_quantization
func.func @pad_per_tensor_quantization(%arg0: tensor<1x2x3x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x7x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1, 2>,
    edge_padding_high = array<i64: 1, 1, 0>,
    interior_padding = array<i64: 0, 0, 1>
  } : (tensor<1x2x3x!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x4x7x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x4x7x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: %[[PAD_0:.*]] = stablehlo.pad %arg0, %arg1
// CHECK: return %[[PAD_0]] : tensor<2x4x7xi8>

// -----

func.func @reduce_per_tensor_quantization(%arg0: tensor<16x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.reduce' that was explicitly marked illegal}}
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {
    dimensions = array<i64: 0>
  } : (tensor<16x!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<!quant.uniform<i8:f32, 1.0:17>>
}

// -----

// CHECK-LABEL: @reduce_per_tensor_precision_quantization
func.func @reduce_per_tensor_precision_quantization(%arg0: tensor<6x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<6x!quant.uniform<i8:f32, 1.0:17>> {
  %output = "stablehlo.reduce_precision"(%arg0) {
    exponent_bits = 5 : i32,
    mantissa_bits = 10 : i32
  } : (tensor<6x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<6x!quant.uniform<i8:f32, 1.0:17>>
  func.return %output : tensor<6x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: stablehlo.reduce_precision {{.*}} : tensor<6xf32>


// -----

func.func @reduce_scatter_per_tensor_quantization(%data: tensor<4x16x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x4x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.reduce_scatter' that was explicitly marked illegal}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<!quant.uniform<i8:f32, 1.0:17>>
    "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<4x4x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<4x4x!quant.uniform<i8:f32, 1.0:17>>
}

// -----

// CHECK-LABEL: @op_reduce_window_per_tensor_quantization
func.func @op_reduce_window_per_tensor_quantization(%arg0: tensor<2x17x31x7x!quant.uniform<i8:f32, 0.1:-30>>, %arg1: tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<2x9x16x7x!quant.uniform<i8:f32, 0.1:-30>> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 0.1:-30>>, %arg3: tensor<!quant.uniform<i8:f32, 0.1:-30>>):
      %1 = "stablehlo.maximum"(%arg2, %arg3) : (tensor<!quant.uniform<i8:f32, 0.1:-30>>, tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<!quant.uniform<i8:f32, 0.1:-30>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> ()
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    window_strides = array<i64: 1, 4, 4, 1>,
    base_dilations = array<i64: 1, 2, 2, 1>,
    window_dilations = array<i64: 1, 2, 2, 1>,
    padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>
  } : (tensor<2x17x31x7x!quant.uniform<i8:f32, 0.1:-30>>, tensor<!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<2x9x16x7x!quant.uniform<i8:f32, 0.1:-30>>
  func.return %0 : tensor<2x9x16x7x!quant.uniform<i8:f32, 0.1:-30>>
}

// CHECK: %[[REDUCE_WINDOW_0:.*]] = "stablehlo.reduce_window"{{.*}}
// CHECK:   (tensor<2x17x31x7xi8>, tensor<i8>) -> tensor<2x9x16x7xi8>

// -----

func.func @reverse_per_tensor_quantization(%operand: tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.reverse' that was explicitly marked illegal}}
  %result = "stablehlo.reverse"(%operand) {
    dimensions = array<i64: 1>
  } : (tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>
  func.return %result : tensor<3x2x!quant.uniform<i8:f32, 0.1:-30>>
}

// -----

// CHECK-LABEL: @round_afz_per_tensor_quantization
func.func @round_afz_per_tensor_quantization(%arg0: tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.round_nearest_afz"(%arg0) {} : (tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: stablehlo.round_nearest_afz {{.*}} : tensor<2xf32>

// -----

// CHECK-LABEL: @round_even_per_tensor_quantization
func.func @round_even_per_tensor_quantization(%arg0: tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>> {
  %0 = "stablehlo.round_nearest_even"(%arg0) {} : (tensor<2x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x!quant.uniform<i8:f32, 1.0:17>>
}

// CHECK: stablehlo.round_nearest_even {{.*}} : tensor<2xf32>

// -----

func.func @scatter_per_tensor_quantization(%arg0: tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<10x2xi32>, %arg2: tensor<10x300x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.scatter' that was explicitly marked illegal}}
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg4: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >
  } : (tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>, tensor<10x2xi32>, tensor<10x300x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<200x100x300x!quant.uniform<i8:f32, 1.0:17>>
}

// -----

// CHECK-LABEL: func.func @select_per_tensor_quantization
func.func @select_per_tensor_quantization(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>, %arg2: tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x3x!quant.uniform<i8:f32, 1.0:17>> {
  // CHECK: %[[SELECT_0:.*]] = stablehlo.select %arg0, %arg1, %arg2 : tensor<2x3xi1>, tensor<2x3xi8>
  // CHECK: return %[[SELECT_0]] : tensor<2x3xi8>

  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<2x3xi1>, tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x3x!quant.uniform<i8:f32, 1.0:17>>
}

// -----

func.func @select_and_scatter_per_tensor_quantization(%arg0: tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<10x23x23x64x!quant.uniform<i8:f32, 1.0:17>>, %arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.select_and_scatter' that was explicitly marked illegal}}
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg4: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.compare"(%arg3, %arg4) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }, {
    ^bb0(%arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg4: tensor<!quant.uniform<i8:f32, 1.0:17>>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<!quant.uniform<i8:f32, 1.0:17>>
      "stablehlo.return"(%1) : (tensor<!quant.uniform<i8:f32, 1.0:17>>) -> ()
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>
  } : (tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>, tensor<10x23x23x64x!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<10x24x24x64x!quant.uniform<i8:f32, 1.0:17>>
}

// -----

func.func @sort_per_tensor_quantization(%input0: tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, %input1: tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>) {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.sort' that was explicitly marked illegal}}
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg2: tensor<!quant.uniform<i8:f32, 1.0:17>>, %arg3: tensor<!quant.uniform<i8:f32, 1.0:17>>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<!quant.uniform<i8:f32, 1.0:17>>, tensor<!quant.uniform<i8:f32, 1.0:17>>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>) -> (tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>)
  func.return %0#0, %0#1: tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>, tensor<16x16x!quant.uniform<i8:f32, 1.0:17>>
}

// -----

// CHECK-LABEL: func.func @slice_per_tensor_qunatization
func.func @slice_per_tensor_qunatization(%arg0: tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.0:17>> {
  // CHECK: %[[SLICE_0:.*]] = stablehlo.slice %arg0 [1:2, 0:4:2] : (tensor<3x4xi8>) -> tensor<1x2xi8>
  // CHECK: return %[[SLICE_0]] : tensor<1x2xi8>
  %0 = "stablehlo.slice"(%arg0) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 2>} : (tensor<3x4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<1x2x!quant.uniform<i8:f32, 1.0:17>>
}

// -----

func.func @while_per_tensor_quantization(%arg0: tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<?x!quant.uniform<i8:f32, 1.0:17>> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.while' that was explicitly marked illegal}}
  %while = "stablehlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<?x!quant.uniform<i8:f32, 1.0:17>>):
    %1 = stablehlo.constant dense<true> : tensor<i1>
    stablehlo.return %1 : tensor<i1>
  },  {
  ^bb0(%arg1: tensor<?x!quant.uniform<i8:f32, 1.0:17>>):
    stablehlo.return %arg1 : tensor<?x!quant.uniform<i8:f32, 1.0:17>>
  }) : (tensor<4x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<?x!quant.uniform<i8:f32, 1.0:17>>
  func.return %while : tensor<?x!quant.uniform<i8:f32, 1.0:17>>
}

// -----

// CHECK-LABEL: func.func @dot_general_with_i8_result_element_type
func.func @dot_general_with_i8_result_element_type(%arg0: tensor<2x3x4x!quant.uniform<i8:f32, 1.0:17>>, %arg1: tensor<2x3x5x!quant.uniform<i8:f32, 1.0:0>>) -> tensor<2x4x5x!quant.uniform<i8:f32, 1.0:17>> {
  // CHECK: stablehlo.dot_general{{.*}} : (tensor<2x3x4xi8>, tensor<2x3x5xi8>) -> tensor<2x4x5xi32>
  // CHECK: %[[CONVERT:.*]] = stablehlo.convert {{.*}} : (tensor<2x4x5xi32>) -> tensor<2x4x5xi8>
  // CHECK: return %[[CONVERT]] : tensor<2x4x5xi8>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x3x4x!quant.uniform<i8:f32, 1.0:17>>, tensor<2x3x5x!quant.uniform<i8:f32, 1.0:0>>) -> tensor<2x4x5x!quant.uniform<i8:f32, 1.0:17>>
  func.return %0 : tensor<2x4x5x!quant.uniform<i8:f32, 1.0:17>>
}

// -----

// CHECK-LABEL: func.func @convolution_with_i8_result_element_type
func.func @convolution_with_i8_result_element_type(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<128x26x26x128x!quant.uniform<i8:f32, 1.000000e+00:5>> {
  // CHECK: stablehlo.convolution{{.*}} : (tensor<128x28x28x1xi8>, tensor<3x3x1x128xi8>) -> tensor<128x26x26x128xi32>
  // CHECK: %[[CONVERT:.*]] = stablehlo.convert {{.*}} : (tensor<128x26x26x128xi32>) -> tensor<128x26x26x128xi8>
  // CHECK: return %[[CONVERT]] : tensor<128x26x26x128xi8>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x26x26x128x!quant.uniform<i8:f32, 1.000000e+00:5>>
  return %0 : tensor<128x26x26x128x!quant.uniform<i8:f32, 1.000000e+00:5>>
}
