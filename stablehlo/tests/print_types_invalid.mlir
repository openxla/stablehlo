// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file

// -----

func.func @unary_eltwise_two_types(%arg0: tensor<?x?xf64>,
                                      %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'stablehlo.abs' 1 operands present, but expected 2}}
  %0 = stablehlo.abs %arg0 : (tensor<?x?xf64>, tensor<?x?xf32>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

// TODO(ajcbik): error message is a bit too strict, should be "compatible" type?
func.func @binary_eltwise_type_mismatch(%arg0: tensor<?x?xf64>,
                                        %arg1: tensor<?x?xf32>) -> tensor<?x?xf64> {
  // expected-error @+1 {{'stablehlo.add' op requires compatible types for all operands and results}}
  %0 = stablehlo.add %arg0, %arg1 : (tensor<?x?xf64>, tensor<?x?xf32>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @binary_eltwise_three_types(%arg0: tensor<?x?xf64>,
                                      %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'stablehlo.add' 2 operands present, but expected 3}}
  %0 = stablehlo.add %arg0, %arg1 : (tensor<?x?xf64>, tensor<?x?xf32>, tensor<?x?xf64>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @binary_eltwise_multiple_out(%arg0: tensor<?x?xf64>,
                                      %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'stablehlo.add' expected single output}}
  %0 = stablehlo.add %arg0, %arg1 : (tensor<?x?xf64>, tensor<?x?xf32>) -> (tensor<?x?xf64>, tensor<?x?xf32>)
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @complex_type_not_type(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{expected non-function type}}
  %0 = stablehlo.complex %arg0, %arg0 : %arg0
  func.return %0 : tensor<1xf64>
}

// -----

func.func @complex_type_not_tensor(%arg0: tensor<1xf64>) -> () {
  // expected-error @+1 {{custom op 'stablehlo.complex' expected tensor with complex element type}}
  %0 = stablehlo.complex %arg0, %arg0 : complex<f64>
  func.return
}

// -----

func.func @complex_type_not_complex(%arg0: tensor<1xf64>) -> () {
  // expected-error @+1 {{custom op 'stablehlo.complex' expected tensor with complex element type}}
  %0 = stablehlo.complex %arg0, %arg0 : tensor<1xf64>
  func.return
}

// -----

func.func @select_type_wrong_type(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>) -> () {
  // expected-error @+1 {{custom op 'stablehlo.select' expected functional type or list of two types}}
  %0 = stablehlo.select %arg0, %arg1, %arg1 : tensor<2x3xi1>
  func.return %0
}

// -----

func.func @select_type_too_many(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>) -> () {
  // expected-error @+1 {{custom op 'stablehlo.select' expected functional type or list of two types}}
  %0 = stablehlo.select %arg0, %arg1, %arg1 : tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>
  func.return
}

// -----

func.func @tuple_type_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{custom op 'stablehlo.tuple' expected tuple type}}
  %0 = stablehlo.tuple %arg0, %arg1 : tensor<1xf64>, tensor<1xf32>
  func.return %0 : tensor<1xf64>
}

// -----

func.func @tuple_type_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{expected '->' in function type}}
  %0 = stablehlo.tuple %arg0, %arg0 : (tensor<1xf64>, tensor<1xf64>)
  func.return %0 : tensor<1xf64>
}

// -----

func.func @tuple_count_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{custom op 'stablehlo.tuple' 2 operands present, but expected 1}}
  %0 = stablehlo.tuple %arg0, %arg0 : tuple<tensor<1xf64>>
  func.return %0 : tensor<1xf64>
}

// -----

func.func @pairwise_count_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{custom op 'stablehlo.optimization_barrier' 2 operands present, but expected 1}}
  %0 = stablehlo.optimization_barrier %arg0, %arg0 : tensor<1xf64>
  func.return %0 : tensor<1xf64>
}

// -----

func.func @pairwise_type_not_list(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+2 {{expected non-function type}}
  // expected-error @+1 {{custom op 'stablehlo.optimization_barrier' expected type list}}
  %0 = stablehlo.optimization_barrier %arg0, %arg0 : %arg0
  func.return %0 : tensor<1xf64>
}

// -----

func.func @one_result_type(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{expected non-function type}}
  %0 = stablehlo.abs %arg0 : %arg0
  func.return %0 : tensor<1xf64>
}

// -----

func.func @reduce_precision_not_literal(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'stablehlo.reduce_precision' expected valid keyword}}
  %0 = stablehlo.reduce_precision %arg0, format = "e2m2" : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @reduce_precision_no_em(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'stablehlo.reduce_precision' expected exponent mantissa in format e#m#, saw z4f2}}
  %0 = stablehlo.reduce_precision %arg0, format = z4f2 : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @reduce_precision_em_order(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'stablehlo.reduce_precision' expected exponent mantissa in format e#m#, saw m2e2}}
  %0 = stablehlo.reduce_precision %arg0, format = m2e2 : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @reduce_precision_no_e_num(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'stablehlo.reduce_precision' expected exponent mantissa in format e#m#, saw em2}}
  %0 = stablehlo.reduce_precision %arg0, format = em2 : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @reduce_precision_overflow_int32_e(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'stablehlo.reduce_precision' unable to parse exponent '2147483648'}}
  %0 = stablehlo.reduce_precision %arg0, format = e2147483648m1 : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @reduce_precision_overflow_int32_m(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'stablehlo.reduce_precision' unable to parse mantissa '2147483648'}}
  %0 = stablehlo.reduce_precision %arg0, format = e1m2147483648 : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}
