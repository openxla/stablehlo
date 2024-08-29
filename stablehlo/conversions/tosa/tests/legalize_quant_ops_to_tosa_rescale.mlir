// RUN: stablehlo-opt %s --stablehlo-quant-legalize-to-tosa-rescale --split-input-file -verify-each | FileCheck %s

// -----
// CHECK-LABEL: @add
func.func @add(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1431655765>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 13>}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>}
  // CHECK: %[[V2:.+]] = stablehlo.add %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]] {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 50>}
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>>
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
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>>
  %0 = "stablehlo.subtract"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}

// -----
// CHECK-LABEL: @mul
func.func @mul(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>}
  // CHECK: %[[V2:.+]] = stablehlo.multiply %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]] {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1717986918>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 37>}
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>>
  %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}

// -----
// CHECK-LABEL: @div
func.func @div(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>> {
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1 {double_round = false, input_zp = -2 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>}
  // CHECK: %[[V2:.+]] = stablehlo.divide %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]] {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1717986918>, output_zp = -3 : i32, per_channel = false, scale32 = true, shift = array<i8: 37>}
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-3>>
  %0 = "stablehlo.divide"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
}

// -----
// CHECK-LABEL: @max
func.func @max(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>> {
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1431655765>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 12>}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1 {double_round = false, input_zp = -2 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 10>}
  // CHECK: %[[V2:.+]] = stablehlo.maximum %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]] {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = -3 : i32, per_channel = false, scale32 = true, shift = array<i8: 51>}
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-3>>
  %0 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
}

// -----
// CHECK-LABEL: @min
func.func @min(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>> {
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1431655765>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 12>}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1 {double_round = false, input_zp = -2 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 10>}
  // CHECK: %[[V2:.+]] = stablehlo.minimum %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]] {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = -3 : i32, per_channel = false, scale32 = true, shift = array<i8: 51>}
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-3>>
  %0 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
}

// -----
// CHECK-LABEL: @abs
func.func @abs(%arg0 : tensor<20x20x!quant.uniform<i8:f32, 0.025:-1>>) -> tensor<20x20x!quant.uniform<i8:f32, 1.5e-01:-128>> {
  // CHECK: %[[V0:.+]] = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>}
  // CHECK: %[[V1:.+]] = stablehlo.abs %[[V0]] : tensor<20x20xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V1]] {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1431655765>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i8: 33>}
  // CHECK: return %[[V3]] : tensor<20x20x!quant.uniform<i8:f32, 1.500000e-01:-128>>
  %0 = "stablehlo.abs"(%arg0) : (tensor<20x20x!quant.uniform<i8:f32, 0.025:-1>>) -> tensor<20x20x!quant.uniform<i8:f32, 1.5e-01:-128>>
  return %0 : tensor<20x20x!quant.uniform<i8:f32, 1.5e-01:-128>>
}

// -----
// CHECK-LABEL: @compareGE
func.func @compareGE(%arg0 : tensor<20x20x!quant.uniform<i8:f32, 0.025:-1>>,
                   %arg1 : tensor<20x20x!quant.uniform<i8:f32, 0.075:-2>>) -> tensor<20x20xi1> {
  // CHECK: %[[V0:.+]] = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1431655765>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 12>}
  // CHECK: %[[V1:.+]] = tosa.rescale %arg1 {double_round = false, input_zp = -2 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 10>}
  // CHECK: %[[V2:.+]] = stablehlo.compare GE, %[[V0]], %[[V1]], TOTALORDER :
  // CHECK: return %[[V2]]
  %0 = stablehlo.compare GE, %arg0, %arg1, TOTALORDER : (tensor<20x20x!quant.uniform<i8:f32, 0.025:-1>>, tensor<20x20x!quant.uniform<i8:f32, 0.075:-2>>) -> tensor<20x20xi1>
  return %0 : tensor<20x20xi1>
}

// -----
// CHECK-LABEL: @compareLT
func.func @compareLT(%arg0 : tensor<20x20x!quant.uniform<i16:f32, 0.025:-1>>,
                   %arg1 : tensor<20x20x!quant.uniform<i16:f32, 0.075:-2>>) -> tensor<20x20xi1> {
  // CHECK: %[[V0:.+]] = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1431655765>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 17>}
  // CHECK: %[[V1:.+]] = tosa.rescale %arg1 {double_round = false, input_zp = -2 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 15>}
  // CHECK: %[[V2:.+]] = stablehlo.compare LT, %[[V0]], %[[V1]], TOTALORDER :
  // CHECK: return %[[V2]]
  %0 = stablehlo.compare LT, %arg0, %arg1, TOTALORDER : (tensor<20x20x!quant.uniform<i16:f32, 0.025:-1>>, tensor<20x20x!quant.uniform<i16:f32, 0.075:-2>>) -> tensor<20x20xi1>
  return %0 : tensor<20x20xi1>
}
