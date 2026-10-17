// RUN: stablehlo-translate --interpret -split-input-file %s
// RUN: stablehlo-opt --stablehlo-legalize-quant-to-math --chlo-legalize-to-stablehlo -split-input-file %s | stablehlo-translate --interpret -split-input-file

func.func @uniform_dequantize_f16() {
  %storage = stablehlo.constant dense<2> : tensor<i8>
  %operand = stablehlo.bitcast_convert %storage : (tensor<i8>) -> tensor<!quant.uniform<i8:f16, 0.5:0>>
  %result = stablehlo.uniform_dequantize %operand : (tensor<!quant.uniform<i8:f16, 0.5:0>>) -> tensor<f16>
  check.expect_eq_const %result, dense<1.0> : tensor<f16>
  func.return
}

// -----

func.func @uniform_dequantize_bf16() {
  %storage = stablehlo.constant dense<2> : tensor<i8>
  %operand = stablehlo.bitcast_convert %storage : (tensor<i8>) -> tensor<!quant.uniform<i8:bf16, 0.5:0>>
  %result = stablehlo.uniform_dequantize %operand : (tensor<!quant.uniform<i8:bf16, 0.5:0>>) -> tensor<bf16>
  check.expect_eq_const %result, dense<1.0> : tensor<bf16>
  func.return
}

// -----

func.func @uniform_dequantize_f64() {
  %storage = stablehlo.constant dense<1> : tensor<i8>
  %operand = stablehlo.bitcast_convert %storage : (tensor<i8>) -> tensor<!quant.uniform<i8:f64, 1.0000000000000002:0>>
  %result = stablehlo.uniform_dequantize %operand : (tensor<!quant.uniform<i8:f64, 1.0000000000000002:0>>) -> tensor<f64>
  check.expect_eq_const %result, dense<1.0000000000000002> : tensor<f64>
  func.return
}

// -----

func.func @uniform_dequantize_per_axis_f16() {
  %storage = stablehlo.constant dense<[[3, 3], [5, 7]]> : tensor<2x2xi8>
  %operand = stablehlo.bitcast_convert %storage : (tensor<2x2xi8>) -> tensor<2x2x!quant.uniform<i8:f16:1, {0.5:1, 0.25:-1}>>
  %result = stablehlo.uniform_dequantize %operand : (tensor<2x2x!quant.uniform<i8:f16:1, {0.5:1, 0.25:-1}>>) -> tensor<2x2xf16>
  check.expect_eq_const %result, dense<[[1.0, 1.0], [2.0, 2.0]]> : tensor<2x2xf16>
  func.return
}

// -----

func.func @uniform_dequantize_per_axis_bf16() {
  %storage = stablehlo.constant dense<[[3, 3], [5, 7]]> : tensor<2x2xi8>
  %operand = stablehlo.bitcast_convert %storage : (tensor<2x2xi8>) -> tensor<2x2x!quant.uniform<i8:bf16:1, {0.5:1, 0.25:-1}>>
  %result = stablehlo.uniform_dequantize %operand : (tensor<2x2x!quant.uniform<i8:bf16:1, {0.5:1, 0.25:-1}>>) -> tensor<2x2xbf16>
  check.expect_eq_const %result, dense<[[1.0, 1.0], [2.0, 2.0]]> : tensor<2x2xbf16>
  func.return
}

// -----

func.func @uniform_dequantize_per_axis_f64() {
  %storage = stablehlo.constant dense<[[3, 3], [5, 7]]> : tensor<2x2xi8>
  %operand = stablehlo.bitcast_convert %storage : (tensor<2x2xi8>) -> tensor<2x2x!quant.uniform<i8:f64:1, {0.5000000000000001:1, 0.25:-1}>>
  %result = stablehlo.uniform_dequantize %operand : (tensor<2x2x!quant.uniform<i8:f64:1, {0.5000000000000001:1, 0.25:-1}>>) -> tensor<2x2xf64>
  check.expect_eq_const %result, dense<[[1.0000000000000002, 1.0], [2.0000000000000004, 2.0]]> : tensor<2x2xf64>
  func.return
}
