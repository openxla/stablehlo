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
