// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file

// expected-error@+1 {{Bounds length is 1, expected to be equal to rank(2) of the tensor}}
func.func @incorrect_bounds_length(%arg0: tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3]>>) -> tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3]>> {
  func.return %arg0 : tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3]>>
}

// -----

// expected-error@+1 {{Bound is 3 for dimension 0 with size 3, expected bound to be dynamic(?)}}
func.func @static_dim_with_bound(%arg0: tensor<3xf32, #stablehlo.type_extensions<bounds = [3]>>) -> tensor<?xf32, #stablehlo.type_extensions<bounds = [3]>> {
  func.return %arg0 : tensor<?xf32, #stablehlo.type_extensions<bounds = [3]>>
}
