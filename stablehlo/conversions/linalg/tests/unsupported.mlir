// RUN: stablehlo-opt %s --stablehlo-legalize-to-linalg | FileCheck %s

// CHECK-LABEL: func.func @bitcast_convert_quantized
func.func @bitcast_convert_quantized(
    %input: tensor<2x2x!quant.uniform<i8:f32, 1.0:0>>) -> tensor<2x2xi8> {
  // CHECK: stablehlo.bitcast_convert
  %result = "stablehlo.bitcast_convert"(%input)
      : (tensor<2x2x!quant.uniform<i8:f32, 1.0:0>>) -> tensor<2x2xi8>
  func.return %result : tensor<2x2xi8>
}

// CHECK-LABEL: func.func @bitcast_convert_rank_changing_quantized
func.func @bitcast_convert_rank_changing_quantized(
    %input: tensor<2x2x!quant.uniform<i8:f32, 1.0:0>>) -> tensor<2xi16> {
  // CHECK: stablehlo.bitcast_convert
  %result = "stablehlo.bitcast_convert"(%input)
      : (tensor<2x2x!quant.uniform<i8:f32, 1.0:0>>) -> tensor<2xi16>
  func.return %result : tensor<2xi16>
}

// CHECK-LABEL: func.func @bitcast_convert_rank_changing_quantized_result
func.func @bitcast_convert_rank_changing_quantized_result(
    %input: tensor<2xi16>) -> tensor<2x2x!quant.uniform<i8:f32, 1.0:0>> {
  // CHECK: stablehlo.bitcast_convert
  %result = "stablehlo.bitcast_convert"(%input)
      : (tensor<2xi16>) -> tensor<2x2x!quant.uniform<i8:f32, 1.0:0>>
  func.return %result : tensor<2x2x!quant.uniform<i8:f32, 1.0:0>>
}
