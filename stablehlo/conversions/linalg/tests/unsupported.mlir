// RUN: stablehlo-opt %s --stablehlo-legalize-to-linalg | FileCheck %s

// CHECK-LABEL: func.func @bitcast_convert_quantized
func.func @bitcast_convert_quantized(
    %input: tensor<2x2x!quant.uniform<i8:f32, 1.0:0>>) -> tensor<2x2xi8> {
  // CHECK: stablehlo.bitcast_convert
  %result = "stablehlo.bitcast_convert"(%input)
      : (tensor<2x2x!quant.uniform<i8:f32, 1.0:0>>) -> tensor<2x2xi8>
  func.return %result : tensor<2x2xi8>
}
