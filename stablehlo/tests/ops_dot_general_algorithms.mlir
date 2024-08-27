// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @dot_algorithm_f8_f8_f32
func.func @dot_algorithm_f8_f8_f32(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = f8E4M3FNUZ,
      rhs_precision_type = f8E4M3FNUZ,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
}

// CHECK-LABEL: func @dot_algorithm_f8_f8_f32_fast_accum
func.func @dot_algorithm_f8_f8_f32_fast_accum(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = f8E4M3FNUZ,
      rhs_precision_type = f8E4M3FNUZ,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = true
    >
  }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
}

// CHECK-LABEL: func @dot_algorithm_f16_f16_f16
func.func @dot_algorithm_f16_f16_f16(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = f16,
      rhs_precision_type = f16,
      accumulation_type = f16,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
}

// CHECK-LABEL: func @dot_algorithm_f16_f16_f32
func.func @dot_algorithm_f16_f16_f32(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = f16,
      rhs_precision_type = f16,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
}

// CHECK-LABEL: func @dot_algorithm_bf16_bf16_bf16
func.func @dot_algorithm_bf16_bf16_bf16(%arg0: tensor<2x2x2xbf16>, %arg1: tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = bf16,
      rhs_precision_type = bf16,
      accumulation_type = bf16,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>  return %0 : tensor<2x2x2xbf16>
}

// CHECK-LABEL: func @dot_algorithm_bf16_bf16_f32
func.func @dot_algorithm_bf16_bf16_f32(%arg0: tensor<2x2x2xbf16>, %arg1: tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = bf16,
      rhs_precision_type = bf16,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>  return %0 : tensor<2x2x2xbf16>
}

// CHECK-LABEL: func @dot_algorithm_bf16_bf16_f32_x3
func.func @dot_algorithm_bf16_bf16_f32_x3(%arg0: tensor<2x2x2xbf16>, %arg1: tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = bf16,
      rhs_precision_type = bf16,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 3,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>  return %0 : tensor<2x2x2xbf16>
}

// CHECK-LABEL: func @dot_algorithm_bf16_bf16_f32_x6
func.func @dot_algorithm_bf16_bf16_f32_x6(%arg0: tensor<2x2x2xbf16>, %arg1: tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = bf16,
      rhs_precision_type = bf16,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 6,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>  return %0 : tensor<2x2x2xbf16>
}

// CHECK-LABEL: func @dot_algorithm_tf32_tf32_f32
func.func @dot_algorithm_tf32_tf32_f32(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = tf32,
      rhs_precision_type = tf32,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
}

// CHECK-LABEL: func @dot_algorithm_tf32_tf32_f32_x3
func.func @dot_algorithm_tf32_tf32_f32_x3(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = tf32,
      rhs_precision_type = tf32,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 3,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
}

// CHECK-LABEL: func @dot_algorithm_f32_f32_f32
func.func @dot_algorithm_f32_f32_f32(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = f32,
      rhs_precision_type = f32,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
}

// CHECK-LABEL: func @dot_algorithm_f64_f64_f64
func.func @dot_algorithm_f64_f64_f64(%arg0: tensor<2x2x2xf64>, %arg1: tensor<2x2x2xf64>) -> tensor<2x2x2xf64> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = f64,
      rhs_precision_type = f64,
      accumulation_type = f64,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xf64>, tensor<2x2x2xf64>) -> tensor<2x2x2xf64>  return %0 : tensor<2x2x2xf64>
}

// -----

func.func @dot_algorithm_f32_f32_f32_l3(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  // expected-error@+4 {{dot algorithm not known to be supported on any hardware: {lhs:'f32', rhs:'f32', accum:'f32', lhs_components:3, rhs_components:1, primitive_ops:1, imprecise:0}}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = f32,
      rhs_precision_type = f32,
      accumulation_type = f32,
      lhs_component_count = 3,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
}

// -----

func.func @dot_algorithm_f32_f32_f32_r3(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  // expected-error@+4 {{dot algorithm not known to be supported on any hardware: {lhs:'f32', rhs:'f32', accum:'f32', lhs_components:1, rhs_components:3, primitive_ops:1, imprecise:0}}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = f32,
      rhs_precision_type = f32,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 3,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
}

// -----

func.func @dot_algorithm_f32_f32_f32_imprecise(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  // expected-error@+4 {{dot algorithm not known to be supported on any hardware: {lhs:'f32', rhs:'f32', accum:'f32', lhs_components:1, rhs_components:1, primitive_ops:1, imprecise:1}}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = f32,
      rhs_precision_type = f32,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = true
    >
  }> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>  return %0 : tensor<2x2x2xf32>
}
