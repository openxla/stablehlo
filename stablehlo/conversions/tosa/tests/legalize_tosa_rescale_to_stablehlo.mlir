// RUN: stablehlo-opt %s --tosa-rescale-legalize-to-stablehlo --split-input-file -verify-each | FileCheck %s

// -----
// CHECK-LABEL: @rescale
func.func @rescale(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>) -> tensor<2x2xi32> {
  %0 = tosa.rescale %arg0 {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1431655765>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 13>} :
            (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>) -> tensor<2x2xi32>

  // convert input quantized type to storage type
  // CHECK-DAG: %[[arg:.+]] = stablehlo.bitcast_convert %arg0 : (tensor<2x2x!quant.uniform<i8:f32, 2.500000e-02:-1>>) -> tensor<2x2xi8>

  // CHECK-DAG: %[[multiplier:.+]] = stablehlo.constant dense<1431655765> : tensor<2x2xi32>
  // CHECK-DAG: %[[shift:.+]] = stablehlo.constant dense<13> : tensor<2x2xi8>
  // CHECK-DAG: %[[input_zp:.+]] = stablehlo.constant dense<-1> : tensor<2x2xi32>
  // CHECK-DAG: %[[output_zp:.+]] = stablehlo.constant dense<0> : tensor<2x2xi32>
  // CHECK-DAG: %[[ones:.+]] = stablehlo.constant dense<1> : tensor<2x2xi64>
  // CHECK-DAG: %[[min:.+]] = stablehlo.constant dense<-2147483648> : tensor<2x2xi64>
  // CHECK-DAG: %[[max:.+]] = stablehlo.constant dense<2147483647> : tensor<2x2xi64>

  // conversions
  // CHECK-DAG: %[[c_multiplier:.+]] = stablehlo.convert %[[multiplier]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_shift:.+]] = stablehlo.convert %[[shift]] : (tensor<2x2xi8>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_input_zp:.+]] = stablehlo.convert %[[input_zp]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_output_zp:.+]] = stablehlo.convert %[[output_zp]] : (tensor<2x2xi32>) -> tensor<2x2xi64>
  // CHECK-DAG: %[[c_value:.+]] = stablehlo.convert %[[arg]] : (tensor<2x2xi8>) -> tensor<2x2xi64>

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
  // CHECK-DAG: %[[result5:.+]] = stablehlo.clamp %[[min]], %[[result4]], %[[max]] : tensor<2x2xi64>
  // CHECK-DAG: %[[result6:.+]] = stablehlo.convert %[[result5]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
  // CHECK: return %[[result6]]

  return %0 : tensor<2x2xi32>
}
