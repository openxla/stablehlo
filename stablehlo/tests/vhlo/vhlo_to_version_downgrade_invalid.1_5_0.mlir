// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.5.0' --verify-diagnostics --split-input-file %s

// expected-error @-3 {{failed to convert VHLO to v1.5.0}}
func.func @dot_general_algorithm(%arg0: tensor<2x2x2xi64>, %arg1: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.dot_general_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
  }> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>  return %0 : tensor<2x2x2xi64>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v1.5.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @none_type() attributes {stablehlo.attr = none } {
  return
}

// -----

// expected-error @-3 {{failed to convert VHLO to v1.5.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @tf32_type() attributes {stablehlo.attr = tf32 } {
  return
}
