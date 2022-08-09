// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file

// -----

func.func @unary_eltwise_wrong_format(%arg0: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'stablehlo.abs' expected function type}}
  %0 = stablehlo.abs(%arg0) : tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @binary_eltwise_wrong_format(%arg0: tensor<?x?xf64>,
                                       %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'stablehlo.add' expected function type}}
  %0 = stablehlo.add(%arg0, %arg1) : tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

func.func @binary_eltwise_wrong_format_sparse(%arg0: tensor<?x?xf64, #CSR>,
                                              %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'stablehlo.add' expected function type}}
  %0 = stablehlo.add(%arg0, %arg1) : tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

// TODO(ajcbik): error message is a bit too strict, should be "compatible" type?
func.func @binary_eltwise_type_mismatch(%arg0: tensor<?x?xf64>,
                                        %arg1: tensor<?x?xf32>) -> tensor<?x?xf64> {
  // expected-error @+1 {{'stablehlo.add' op requires compatible types for all operands and results}}
  %0 = stablehlo.add(%arg0, %arg1) : (tensor<?x?xf64>, tensor<?x?xf32>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @binary_eltwise_three_types(%arg0: tensor<?x?xf64>,
                                      %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'stablehlo.add' 2 operands present, but expected 3}}
  %0 = stablehlo.add(%arg0, %arg1) : (tensor<?x?xf64>, tensor<?x?xf32>, tensor<?x?xf64>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

