// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @uniform_quantize() {
  %operand = stablehlo.constant dense<[4.0, 15.0]> : tensor<2xf32>
  %result = "stablehlo.uniform_quantize"(%operand) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>
  %bitcast_result = "stablehlo.bitcast_convert"(%result) : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>) ->  tensor<2xi8>
  check.expect_eq_const %bitcast_result, dense<[10, 10]> : tensor<2xi8>
  func.return
}

// -----

func.func @uniform_quantize() {
  %operand = stablehlo.constant dense<[10, 10]> : tensor<2xi8>
  %bitcast_operand = "stablehlo.bitcast_convert"(%operand) : (tensor<2xi8>) ->  tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>
  %result = "stablehlo.uniform_quantize"(%bitcast_operand) : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>) -> tensor<2x!quant.uniform<i8:f32:0, {0.1:-20,0.2:-30}>>
  %bitcast_result = "stablehlo.bitcast_convert"(%result) : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-20,0.2:-30}>>) ->  tensor<2xi8>
  check.expect_eq_const %bitcast_result, dense<[20, 45]> : tensor<2xi8>
  func.return
}

// -----

func.func @uniform_dequantize() {
  %operand = stablehlo.constant dense<[10, 10]> : tensor<2xi8>
  %bitcast_operand = "stablehlo.bitcast_convert"(%operand) : (tensor<2xi8>) ->  tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>
  %result = "stablehlo.uniform_dequantize"(%bitcast_operand) : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>) -> tensor<2xf32>
  check.expect_almost_eq_const %result, dense<[4.0, 15.0]> : tensor<2xf32>
  func.return
}

// -----

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


// -----

func.func @uniform_qdq() {
  %operand = stablehlo.constant dense<[4.0, 15.0]> : tensor<2xf32>
  %quantize = "stablehlo.uniform_quantize"(%operand) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>
  %result = "stablehlo.uniform_dequantize"(%quantize) : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>) -> tensor<2xf32>
  check.expect_almost_eq_const %result, dense<[4.0, 15.0]> : tensor<2xf32>
  func.return
}

// -----

func.func @quantized_add() {
  %operand1 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %operand2 = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf32>
  %q_operand1 = "stablehlo.uniform_quantize"(%operand1) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.1:-30>>
  %q_operand2 = "stablehlo.uniform_quantize"(%operand2) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.5:-20>>
  %result = "stablehlo.add"(%q_operand1, %q_operand2) : (tensor<2x!quant.uniform<i8:f32, 0.1:-30>>, tensor<2x!quant.uniform<i8:f32, 0.5:-20>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:-20>>
  %bitcast_result = "stablehlo.bitcast_convert"(%result) : (tensor<2x!quant.uniform<i8:f32, 0.5:-20>>) ->  tensor<2xi8>
  check.expect_eq_const %bitcast_result, dense<[-12, -8]> : tensor<2xi8>
  func.return
}
