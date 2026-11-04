// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.20.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.20.0}}
module {
  func.func @dot_general_algorithm_f8e4m3fn_x3(%arg0: tensor<8x8x16xbf16>, %arg1: tensor<8x16x8xbf16>) -> tensor<8x8x8xf32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.dot_general_v2' that was explicitly marked illegal}}
    %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
      algorithm = #stablehlo.dot_algorithm<
        lhs_precision_type = f8E4M3FN,
        rhs_precision_type = f8E4M3FN,
        accumulation_type = f32,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 3,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<8x8x16xbf16>, tensor<8x16x8xbf16>) -> tensor<8x8x8xf32>
    func.return %0 : tensor<8x8x8xf32>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.20.0}}
module {
  func.func @dot_general_algorithm_f8e4m3fn_x4(%arg0: tensor<8x8x16xbf16>, %arg1: tensor<8x16x8xbf16>) -> tensor<8x8x8xf32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.dot_general_v2' that was explicitly marked illegal}}
    %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
      dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
      algorithm = #stablehlo.dot_algorithm<
        lhs_precision_type = f8E4M3FN,
        rhs_precision_type = f8E4M3FN,
        accumulation_type = f32,
        lhs_component_count = 1,
        rhs_component_count = 1,
        num_primitive_operations = 4,
        allow_imprecise_accumulation = false
      >
    }> : (tensor<8x8x16xbf16>, tensor<8x16x8xbf16>) -> tensor<8x8x8xf32>
    func.return %0 : tensor<8x8x8xf32>
  }
}
