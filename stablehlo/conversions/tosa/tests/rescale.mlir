// RUN: stablehlo-opt %s --tosa-rescale-legalize-to-stablehlo --split-input-file -verify-each | FileCheck %s

// -----
// CHECK-LABEL: @rescale1
func.func @rescale1(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>) -> tensor<2x2xi32> {
  %0 = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1431655765>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 13>} :
            (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>) -> tensor<2x2xi32>

  // convert input quantized type to storage type
  // CHECK-DAG: %[[arg:.+]] = stablehlo.bitcast_convert %arg0 : (tensor<2x2x!quant.uniform<i8:f32, 2.500000e-02:-1>>) -> tensor<2x2xi8>

  // CHECK-DAG: %[[multiplier:.+]] = stablehlo.constant dense<1431655765> : tensor<2x2xi32>
  // CHECK-DAG: %[[shift:.+]] = stablehlo.constant dense<13> : tensor<2x2xi8>
  // CHECK-DAG: %[[input_zp:.+]] = stablehlo.constant dense<-1> : tensor<2x2xi32>
  // CHECK-DAG: %[[output_zp:.+]] = stablehlo.constant dense<0> : tensor<2x2xi32>
  // CHECK-DAG: %[[ones:.+]] = stablehlo.constant dense<1> : tensor<2x2xi64>
  // CHECK-DAG: %[[min:.+]] = stablehlo.constant dense<-2147483648> : tensor<2x2xi32>
  // CHECK-DAG: %[[max:.+]] = stablehlo.constant dense<2147483647> : tensor<2x2xi32>

  // conversions
  // CHECK-DAG: %[[c_multiplier:.+]] = stablehlo.convert %[[multiplier]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_shift:.+]] = stablehlo.convert %[[shift]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_input_zp:.+]] = stablehlo.convert %[[input_zp]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_output_zp:.+]] = stablehlo.convert %[[output_zp]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_value:.+]] = stablehlo.convert %[[arg]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_max:.+]] = stablehlo.convert %[[max]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_min:.+]] = stablehlo.convert %[[min]] : (tensor<2x2xi32>) -> tensor<2x2xi64>

  // value - input_zp
  // CHECK-DAG: %[[value:.+]] = stablehlo.subtract %[[c_value]], %[[c_input_zp]] : tensor<2x2xi64>
  // (shift - 1)
  // CHECK-DAG: %[[adjusted_shift:.+]] = stablehlo.subtract %[[c_shift]], %[[ones]] : tensor<2x2xi64>
  // 1 << (shift -1)
  // CHECK-DAG: %[[round:.+]] = stablehlo.shift_left %[[ones]], %[[adjusted_shift]] : tensor<2x2xi64>
  // value * multiplier
  // CHECK-DAG: %[[result1:.+]] = stablehlo.multiply %[[value]], %[[c_multiplier]] : tensor<2x2xi64>
  // value * multiplier + round
  // CHECK-DAG: %[[result2:.+]] = stablehlo.add %[[result1]], %[[round]] : tensor<2x2xi64>
  // (value * multiplier + round) >> c_shift
  // CHECK-DAG: %[[result3:.+]] = stablehlo.shift_right_arithmetic %[[result2]], %[[c_shift]] : tensor<2x2xi64>
  // (value * multiplier + round) >> c_shift + output_zp
  // CHECK-DAG: %[[result4:.+]] = stablehlo.add %[[result3]], %[[c_output_zp]] : tensor<2x2xi64>
  // clamp to destination type
  // CHECK-DAG: %[[result5:.+]] = stablehlo.clamp %[[c_min]], %[[result4]], %[[c_max]] : tensor<2x2xi64>
  // CHECK-DAG: %[[result6:.+]] = stablehlo.convert %[[result5]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
  // CHECK: return %[[result6]]

  return %0 : tensor<2x2xi32>
}

// -----
// CHECK-LABEL: @rescale2
func.func @rescale2(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2xi32> {
  %0 = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>} :
            (tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2xi32>

  // convert input quantized type to storage type
  // CHECK-DAG: %[[arg:.+]] = stablehlo.bitcast_convert %arg0 : (tensor<2x2x!quant.uniform<i8:f32, 0.074999999999999997:-1>>) -> tensor<2x2xi8>

  // CHECK-DAG: %[[multiplier:.+]] = stablehlo.constant dense<1073741824> : tensor<2x2xi32>
  // CHECK-DAG: %[[shift:.+]] = stablehlo.constant dense<11> : tensor<2x2xi8>
  // CHECK-DAG: %[[input_zp:.+]] = stablehlo.constant dense<-1> : tensor<2x2xi32>
  // CHECK-DAG: %[[output_zp:.+]] = stablehlo.constant dense<0> : tensor<2x2xi32>
  // CHECK-DAG: %[[ones:.+]] = stablehlo.constant dense<1> : tensor<2x2xi64>
  // CHECK-DAG: %[[min:.+]] = stablehlo.constant dense<-2147483648> : tensor<2x2xi32>
  // CHECK-DAG: %[[max:.+]] = stablehlo.constant dense<2147483647> : tensor<2x2xi32>

  // conversions
  // CHECK-DAG: %[[c_multiplier:.+]] = stablehlo.convert %[[multiplier]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_shift:.+]] = stablehlo.convert %[[shift]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_input_zp:.+]] = stablehlo.convert %[[input_zp]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_output_zp:.+]] = stablehlo.convert %[[output_zp]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_value:.+]] = stablehlo.convert %[[arg]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_max:.+]] = stablehlo.convert %[[max]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_min:.+]] = stablehlo.convert %[[min]] : (tensor<2x2xi32>) -> tensor<2x2xi64>

  // value - input_zp
  // CHECK-DAG: %[[value:.+]] = stablehlo.subtract %[[c_value]], %[[c_input_zp]] : tensor<2x2xi64>
  // (shift - 1)
  // CHECK-DAG: %[[adjusted_shift:.+]] = stablehlo.subtract %[[c_shift]], %[[ones]] : tensor<2x2xi64>
  // 1 << (shift -1)
  // CHECK-DAG: %[[round:.+]] = stablehlo.shift_left %[[ones]], %[[adjusted_shift]] : tensor<2x2xi64>
  // value * multiplier
  // CHECK-DAG: %[[result1:.+]] = stablehlo.multiply %[[value]], %[[c_multiplier]] : tensor<2x2xi64>
  // value * multiplier + round
  // CHECK-DAG: %[[result2:.+]] = stablehlo.add %[[result1]], %[[round]] : tensor<2x2xi64>
  // (value * multiplier + round) >> c_shift
  // CHECK-DAG: %[[result3:.+]] = stablehlo.shift_right_arithmetic %[[result2]], %[[c_shift]] : tensor<2x2xi64>
  // (value * multiplier + round) >> c_shift + output_zp
  // CHECK-DAG: %[[result4:.+]] = stablehlo.add %[[result3]], %[[c_output_zp]] : tensor<2x2xi64>
  // clamp to destination type
  // CHECK-DAG: %[[result5:.+]] = stablehlo.clamp %[[c_min]], %[[result4]], %[[c_max]] : tensor<2x2xi64>
  // CHECK-DAG: %[[result6:.+]] = stablehlo.convert %[[result5]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
  // CHECK: return %[[result6]]

  return %0 : tensor<2x2xi32>
}


// -----
// CHECK-LABEL: @rescale3
func.func @rescale3(%arg0 : tensor<2x2xi32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  %0 = tosa.rescale %arg0 {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 50>} :
            (tensor<2x2xi32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>

  // input is not quantized type, no bitcast_convert of input

  // CHECK-DAG: %[[multiplier:.+]] = stablehlo.constant dense<1073741824> : tensor<2x2xi32>
  // CHECK-DAG: %[[shift:.+]] = stablehlo.constant dense<50> : tensor<2x2xi8>
  // CHECK-DAG: %[[input_zp:.+]] = stablehlo.constant dense<0> : tensor<2x2xi32>
  // CHECK-DAG: %[[output_zp:.+]] = stablehlo.constant dense<-1> : tensor<2x2xi32>
  // CHECK-DAG: %[[ones:.+]] = stablehlo.constant dense<1> : tensor<2x2xi64>
  // CHECK-DAG: %[[min:.+]] = stablehlo.constant dense<-128> : tensor<2x2xi32>
  // CHECK-DAG: %[[max:.+]] = stablehlo.constant dense<127> : tensor<2x2xi32>

  // conversions
  // CHECK-DAG: %[[c_multiplier:.+]] = stablehlo.convert %[[multiplier]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_shift:.+]] = stablehlo.convert %[[shift]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_input_zp:.+]] = stablehlo.convert %[[input_zp]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_output_zp:.+]] = stablehlo.convert %[[output_zp]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_value:.+]] = stablehlo.convert %arg0 : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_max:.+]] = stablehlo.convert %[[max]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_min:.+]] = stablehlo.convert %[[min]] : (tensor<2x2xi32>) -> tensor<2x2xi64>

  // value - input_zp
  // CHECK-DAG: %[[value:.+]] = stablehlo.subtract %[[c_value]], %[[c_input_zp]] : tensor<2x2xi64>
  // (shift - 1)
  // CHECK-DAG: %[[adjusted_shift:.+]] = stablehlo.subtract %[[c_shift]], %[[ones]] : tensor<2x2xi64>
  // 1 << (shift -1)
  // CHECK-DAG: %[[round:.+]] = stablehlo.shift_left %[[ones]], %[[adjusted_shift]] : tensor<2x2xi64>
  // value * multiplier
  // CHECK-DAG: %[[result1:.+]] = stablehlo.multiply %[[value]], %[[c_multiplier]] : tensor<2x2xi64>
  // value * multiplier + round
  // CHECK-DAG: %[[result2:.+]] = stablehlo.add %[[result1]], %[[round]] : tensor<2x2xi64>
  // (value * multiplier + round) >> c_shift
  // CHECK-DAG: %[[result3:.+]] = stablehlo.shift_right_arithmetic %[[result2]], %[[c_shift]] : tensor<2x2xi64>
  // (value * multiplier + round) >> c_shift + output_zp
  // CHECK-DAG: %[[result4:.+]] = stablehlo.add %[[result3]], %[[c_output_zp]] : tensor<2x2xi64>
  // clamp to destination type
  // CHECK-DAG: %[[result5:.+]] = stablehlo.clamp %[[c_min]], %[[result4]], %[[c_max]] : tensor<2x2xi64>
  // CHECK-DAG: %[[result6:.+]] = stablehlo.convert %[[result5]] : (tensor<2x2xi64>) -> tensor<2x2xi8>

  // bitcast convert back to quantized output type
  // CHECK: %[[result:.+]] = stablehlo.bitcast_convert %[[result6]] : (tensor<2x2xi8>) -> tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>>
  // CHECK: return %[[result]]

  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}
