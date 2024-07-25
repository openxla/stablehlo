// RUN: stablehlo-opt %s --stablehlo-quant-legalize-to-tosa-rescale --split-input-file -verify-each | FileCheck %s

// -----
// CHECK-LABEL: @add
func.func @add(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1431655765>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 13>}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>}
  // CHECK: %[[V2:.+]] = stablehlo.add %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]] {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 50>}
  // CHECK: return %[[V3]]
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}

// -----
// CHECK-LABEL: @sub
func.func @sub(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1431655765>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 13>}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>}
  // CHECK: %[[V2:.+]] = stablehlo.subtract %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]] {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 50>}
  // CHECK: return %[[V3]]
  %0 = "stablehlo.subtract"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}
