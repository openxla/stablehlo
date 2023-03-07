// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --mlir-print-op-generic --split-input-file %s | FileCheck %s
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)
// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo -emit-bytecode -debug-only=vhlo-bytecode %s 2>&1 | (! grep 'Not Implemented')
// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo -emit-bytecode %s | stablehlo-opt -debug-only=vhlo-bytecode 2>&1 | (! grep 'Not Implemented')

// ============ ATTRIBUTES ============

func.func @attr_comparison_direction_eq(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #vhlo<comparison_direction_v1 EQ>
    comparison_direction = #stablehlo<comparison_direction EQ>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_eq"

func.func @attr_comparison_direction_ne(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #vhlo<comparison_direction_v1 NE>
    comparison_direction = #stablehlo<comparison_direction NE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_ne"

func.func @attr_comparison_direction_ge(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #vhlo<comparison_direction_v1 GE>
    comparison_direction = #stablehlo<comparison_direction GE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_ge"

func.func @attr_comparison_direction_gt(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #vhlo<comparison_direction_v1 GT>
    comparison_direction = #stablehlo<comparison_direction GT>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_gt"

func.func @attr_comparison_direction_le(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #vhlo<comparison_direction_v1 LE>
    comparison_direction = #stablehlo<comparison_direction LE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_le"

func.func @attr_comparison_direction_lt(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #vhlo<comparison_direction_v1 LT>
    comparison_direction = #stablehlo<comparison_direction LT>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_lt"

func.func @attr_comparison_type_notype(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>
    // CHECK: compare_type = #vhlo<comparison_type_v1 NOTYPE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_type_notype"

func.func @attr_comparison_type_float(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    // CHECK: compare_type = #vhlo<comparison_type_v1 FLOAT>,
    compare_type = #stablehlo<comparison_type FLOAT>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_type_float"

func.func @attr_comparison_type_totalorder(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    // CHECK: compare_type = #vhlo<comparison_type_v1 TOTALORDER>,
    compare_type = #stablehlo<comparison_type TOTALORDER>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_type_totalorder"

func.func @attr_comparison_type_signed(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    // CHECK: compare_type = #vhlo<comparison_type_v1 SIGNED>,
    compare_type = #stablehlo<comparison_type SIGNED>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_type_signed"

func.func @attr_comparison_type_unsigned(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    // CHECK: compare_type = #vhlo<comparison_type_v1 UNSIGNED>,
    compare_type = #stablehlo<comparison_type UNSIGNED>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_type_unsigned"

// ConvDimensionNumbers aka #stablehlo.conv is covered below.

func.func @attr_custom_call_api_version_unspecified(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = #vhlo<api_version_v1 API_VERSION_UNSPECIFIED>
    api_version = 0 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_custom_call_api_version_unspecified"

func.func @attr_custom_call_api_version_original(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = #vhlo<api_version_v1 API_VERSION_ORIGINAL>
    api_version = 1 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_custom_call_api_version_original"

func.func @attr_custom_call_api_version_status_returning(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = #vhlo<api_version_v1 API_VERSION_STATUS_RETURNING>
    api_version = 2 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_custom_call_api_version_status_returning"

func.func @attr_custom_call_api_version_status_returning_unified(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = #vhlo<api_version_v1 API_VERSION_STATUS_RETURNING_UNIFIED>
    api_version = 3 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_custom_call_api_version_status_returning_unified"

// DotDimensionNumbers aka #stablehlo.dot is covered below.

func.func @attr_fft_type_fft(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
    // CHECK: fft_type = #vhlo<fft_type_v1 FFT>
    fft_type = #stablehlo<fft_type FFT>,
    fft_length = dense<16> : tensor<1xi64>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}
// CHECK-LABEL: "attr_fft_type_fft"

func.func @attr_fft_type_ifft(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
    // CHECK: fft_type = #vhlo<fft_type_v1 IFFT>
    fft_type = #stablehlo<fft_type IFFT>,
    fft_length = dense<16> : tensor<1xi64>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}
// CHECK-LABEL: "attr_fft_type_ifft"

func.func @attr_fft_type_rfft(%arg0: tensor<16xf32>) -> tensor<9xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
    // CHECK: fft_type = #vhlo<fft_type_v1 RFFT>
    fft_type = #stablehlo<fft_type RFFT>,
    fft_length = dense<16> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<9xcomplex<f32>>
  func.return %0 : tensor<9xcomplex<f32>>
}
// CHECK-LABEL: "attr_fft_type_rfft"

func.func @attr_fft_type_irfft(%arg0: tensor<9xcomplex<f32>>) -> tensor<16xf32> {
  %0 = "stablehlo.fft"(%arg0) {
    // CHECK: fft_type = #vhlo<fft_type_v1 IRFFT>
    fft_type = #stablehlo<fft_type IRFFT>,
    fft_length = dense<16> : tensor<1xi64>
  } : (tensor<9xcomplex<f32>>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "attr_fft_type_irfft"

// GatherDimensionNumbers aka #stablehlo.gather is covered below.

func.func @attr_precision_config_default(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {
    // CHECK: precision_config = #vhlo.array_v1<[#vhlo<precision_v1 DEFAULT>, #vhlo<precision_v1 DEFAULT>]>
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: "attr_precision_config_default"

func.func @attr_precision_config_high(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {
    // CHECK: precision_config = #vhlo.array_v1<[#vhlo<precision_v1 HIGH>, #vhlo<precision_v1 HIGH>]>
    precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: "attr_precision_config_high"

func.func @attr_precision_config_highest(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {
    // CHECK: precision_config = #vhlo.array_v1<[#vhlo<precision_v1 HIGHEST>, #vhlo<precision_v1 HIGHEST>]>
    precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: "attr_precision_config_highest"

func.func @attr_rng_algorithm_default(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = "stablehlo.rng_bit_generator"(%arg0) {
    // CHECK: rng_algorithm = #vhlo<rng_algorithm_v1 DEFAULT>
    rng_algorithm = #stablehlo<rng_algorithm DEFAULT>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}
// CHECK-LABEL: "attr_rng_algorithm_default"

func.func @attr_rng_algorithm_three_fry(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = "stablehlo.rng_bit_generator"(%arg0) {
    // CHECK: rng_algorithm = #vhlo<rng_algorithm_v1 THREE_FRY>
    rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}
// CHECK-LABEL: "attr_rng_algorithm_three_fry"

func.func @attr_rng_algorithm_philox(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = "stablehlo.rng_bit_generator"(%arg0) {
    // CHECK: rng_algorithm = #vhlo<rng_algorithm_v1 PHILOX>
    rng_algorithm = #stablehlo<rng_algorithm PHILOX>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}
// CHECK-LABEL: "attr_rng_algorithm_philox"

func.func @attr_rng_distribution_uniform(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<?xindex>) -> tensor<f32> {
  %0 = "stablehlo.rng"(%arg0, %arg1, %arg2) {
    // CHECK: rng_distribution = #vhlo<rng_distribution_v1 UNIFORM>
    rng_distribution = #stablehlo<rng_distribution UNIFORM>
  } : (tensor<f32>, tensor<f32>, tensor<?xindex>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_rng_distribution_uniform"

func.func @attr_rng_distribution_normal(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<?xindex>) -> tensor<f32> {
  %0 = "stablehlo.rng"(%arg0, %arg1, %arg2) {
    // CHECK: rng_distribution = #vhlo<rng_distribution_v1 NORMAL>
    rng_distribution = #stablehlo<rng_distribution NORMAL>
  } : (tensor<f32>, tensor<f32>, tensor<?xindex>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_rng_distribution_normal"

// ScatterDimensionNumbers aka #stablehlo.scatter is covered below.

func.func @attr_transpose_no_transpose(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    // transpose_a = #vhlo<transpose_v1 NO_TRANSPOSE>,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "attr_transpose_no_transpose"

func.func @attr_transpose_transpose(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    // transpose_a = #vhlo<transpose_v1 TRANSPOSE>,
    transpose_a = #stablehlo<transpose TRANSPOSE>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "attr_transpose_transpose"

func.func @attr_transpose_adjoint(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    // transpose_a = #vhlo<transpose_v1 ADJOINT>,
    transpose_a = #stablehlo<transpose ADJOINT>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "attr_transpose_adjoint"

// TypeExtensionsAttr aka #stablehlo.type_extensions is covered below.

func.func @attr_type_extensions_bounds(
    %arg0: tensor<?x?xf32, #stablehlo.type_extensions<bounds = [16, ?]>>)
    -> tensor<?x?xf32, #stablehlo.type_extensions<bounds = [16, ?]>> {
  // CHECK: "vhlo.return_v1"(%arg0) : (!vhlo.tensor_v1<?x?x!vhlo.f32_v1, #vhlo.type_extensions_v1<bounds = [16, ?]>>) -> ()
  func.return %arg0 : tensor<?x?xf32, #stablehlo.type_extensions<bounds = [16, ?]>>
}

// CHECK-LABEL: "attr_type_extensions_bounds"

// ============ DEFAULTS ============

func.func @default_all_gather(%arg0: tensor<16x8xf32>) -> tensor<16x16xf32> {
  //               CHECK: "vhlo.all_gather_v1"(%arg0) {
  //          CHECK-SAME:   all_gather_dim = #vhlo.integer_v1<1 : i64>
  //          CHECK-SAME:   channel_id = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME{LITERAL}:   replica_groups = #vhlo.tensor_v1<dense<[[0], [1]]> : tensor<2x1xi64>>,
  //          CHECK-SAME:   use_global_device_ids = #vhlo.bool_v1<false>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x16x!vhlo.f32_v1>
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  } : (tensor<16x8xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "default_all_gather"

func.func @default_all_reduce(%arg0: tensor<f32>) -> tensor<f32> {
  //               CHECK: "vhlo.all_reduce_v1"(%arg0) ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG2:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "vhlo.add_v1"(%[[ARG1]], %[[ARG2]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  //          CHECK-NEXT:     "vhlo.return_v1"(%[[VAL1]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  //          CHECK-NEXT: }) {
  //          CHECK-SAME:   channel_id = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME{LITERAL}:   replica_groups = #vhlo.tensor_v1<dense<[[0], [1]]> : tensor<2x1xi64>>,
  //          CHECK-SAME:   use_global_device_ids = #vhlo.bool_v1<false>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "default_all_reduce"

func.func @default_all_to_all(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
  //               CHECK: "vhlo.all_to_all_v1"(%arg0) {
  //          CHECK-SAME:   channel_id = #vhlo.integer_v1<0 : i64>,
  //          CHECK-SAME:   concat_dimension = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME{LITERAL}:   replica_groups = #vhlo.tensor_v1<dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>>,
  //          CHECK-SAME:   split_count = #vhlo.integer_v1<4 : i64>
  //          CHECK-SAME:   split_dimension = #vhlo.integer_v1<1 : i64>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<4x16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x4x!vhlo.f32_v1>
  %0 = "stablehlo.all_to_all"(%arg0) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}
// CHECK-LABEL: "default_all_to_all"

func.func @default_cholesky(%arg0: tensor<1x16x16xf32>) -> tensor<1x16x16xf32> {
  //      CHECK: "vhlo.cholesky_v1"(%arg0) {
  // CHECK-SAME:   lower = #vhlo.bool_v1<false>
  // CHECK-SAME: } : (!vhlo.tensor_v1<1x16x16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<1x16x16x!vhlo.f32_v1>
  %0 = "stablehlo.cholesky"(%arg0) : (tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
  func.return %0 : tensor<1x16x16xf32>
}
// CHECK-LABEL: "default_cholesky"

func.func @default_collective_permute(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  //               CHECK: "vhlo.collective_permute_v1"(%arg0) {
  //          CHECK-SAME:   channel_id = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME{LITERAL}:   source_target_pairs = #vhlo.tensor_v1<dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x8x!vhlo.f32_v1>
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}
// CHECK-LABEL: "default_collective_permute"

func.func @default_compare(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  //      CHECK: "vhlo.compare_v1"(%arg0, %arg1) {
  // CHECK-SAME:   compare_type = #vhlo<comparison_type_v1 NOTYPE>,
  // CHECK-SAME:   comparison_direction = #vhlo<comparison_direction_v1 EQ>
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "default_compare"

func.func @default_convolution(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>) -> tensor<1x6x6x16xf32> {
  //      CHECK: "vhlo.convolution_v1"(%arg0, %arg1) {
  // CHECK-SAME:   batch_group_count = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME:   feature_group_count = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME:   input_batch_dimension = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   input_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   input_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
  // CHECK-SAME:   kernel_input_feature_dimension = #vhlo.integer_v1<2 : i64>,
  // CHECK-SAME:   kernel_output_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   kernel_spatial_dimensions = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   lhs_dilation = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>,
  // CHECK-SAME:   output_batch_dimension = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   output_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   output_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
  // CHECK-SAME:   padding = #vhlo.tensor_v1<dense<0> : tensor<2x2xi64>>,
  // CHECK-SAME:   precision_config = #vhlo.array_v1<[#vhlo<precision_v1 DEFAULT>, #vhlo<precision_v1 DEFAULT>]>,
  // CHECK-SAME:   rhs_dilation = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>,
  // CHECK-SAME:   window_reversal = #vhlo.tensor_v1<dense<false> : tensor<2xi1>>,
  // CHECK-SAME:   window_strides = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<1x8x8x207x!vhlo.f32_v1>, !vhlo.tensor_v1<3x3x207x16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<1x6x6x16x!vhlo.f32_v1>
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %0 : tensor<1x6x6x16xf32>
}
// CHECK-LABEL: "default_convolution"

func.func @default_custom_call(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "vhlo.custom_call_v1"(%arg0) {
  // CHECK-SAME:   api_version = #vhlo<api_version_v1 API_VERSION_ORIGINAL>,
  // CHECK-SAME:   backend_config = #vhlo.string_v1<"">,
  // CHECK-SAME:   call_target_name = #vhlo.string_v1<"foo">,
  // CHECK-SAME:   called_computations = #vhlo.array_v1<[]>,
  // CHECK-SAME:   has_side_effect = #vhlo.bool_v1<false>,
  // CHECK-SAME:   operand_layouts = #vhlo.array_v1<[]>,
  // CHECK-SAME:   output_operand_aliases = #vhlo.array_v1<[]>
  // CHECK-SAME:   result_layouts = #vhlo.array_v1<[]>
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo"
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "default_custom_call"

func.func @default_dot_general(%arg0: tensor<8x8x16xf32>, %arg1: tensor<8x16x8xf32>) -> tensor<8x8x8xf32> {
  //      CHECK: "vhlo.dot_general_v1"(%arg0, %arg1) {
  // CHECK-SAME:   lhs_batching_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
  // CHECK-SAME:   lhs_contracting_dimensions = #vhlo.tensor_v1<dense<2> : tensor<1xi64>>,
  // CHECK-SAME:   precision_config = #vhlo.array_v1<[#vhlo<precision_v1 DEFAULT>, #vhlo<precision_v1 DEFAULT>]>,
  // CHECK-SAME:   rhs_batching_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
  // CHECK-SAME:   rhs_contracting_dimensions = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<8x8x16x!vhlo.f32_v1>, !vhlo.tensor_v1<8x16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<8x8x8x!vhlo.f32_v1>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<8x8x16xf32>, tensor<8x16x8xf32>) -> tensor<8x8x8xf32>
  func.return %0 : tensor<8x8x8xf32>
}
// CHECK-LABEL: "default_dot_general"

func.func @default_dot(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  //      CHECK: "vhlo.dot_v1"(%arg0, %arg1) {
  // CHECK-SAME:   precision_config = #vhlo.array_v1<[#vhlo<precision_v1 DEFAULT>, #vhlo<precision_v1 DEFAULT>]>
  // CHECK-SAME: } : (!vhlo.tensor_v1<8x16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<8x8x!vhlo.f32_v1>
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: "default_dot"

func.func @default_dynamic_broadcast_in_dim(%arg0: tensor<?x?xf32>, %arg1: tensor<2xindex>) -> tensor<?x?xf32> {
  //      CHECK: "vhlo.dynamic_broadcast_in_dim_v1"(%arg0, %arg1) {
  // CHECK-SAME:   broadcast_dimensions = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   known_expanding_dimensions = #vhlo.tensor_v1<dense<> : tensor<0xi64>>,
  // CHECK-SAME:   known_nonexpanding_dimensions = #vhlo.tensor_v1<dense<> : tensor<0xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<?x?x!vhlo.f32_v1>, !vhlo.tensor_v1<2x!vhlo.index_v1>) -> !vhlo.tensor_v1<?x?x!vhlo.f32_v1>
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: "default_dynamic_broadcast_in_dim"

func.func @default_dynamic_conv(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>, %arg2: tensor<4xi32>) -> tensor<1x?x?x16xf32> {
  //      CHECK: "vhlo.dynamic_conv_v1"(%arg0, %arg1, %arg2) {
  // CHECK-SAME:   batch_group_count = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME:   feature_group_count = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME:   input_batch_dimension = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   input_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   input_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
  // CHECK-SAME:   kernel_input_feature_dimension = #vhlo.integer_v1<2 : i64>,
  // CHECK-SAME:   kernel_output_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   kernel_spatial_dimensions = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   lhs_dilation = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>,
  // CHECK-SAME:   output_batch_dimension = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   output_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   output_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
  // CHECK-SAME:   padding = #vhlo.tensor_v1<dense<0> : tensor<2x2xi64>>,
  // CHECK-SAME:   precision_config = #vhlo.array_v1<[#vhlo<precision_v1 DEFAULT>, #vhlo<precision_v1 DEFAULT>]>,
  // CHECK-SAME:   rhs_dilation = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>,
  // CHECK-SAME:   window_reversal = #vhlo.tensor_v1<dense<false> : tensor<2xi1>>,
  // CHECK-SAME:   window_strides = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<1x8x8x207x!vhlo.f32_v1>, !vhlo.tensor_v1<3x3x207x16x!vhlo.f32_v1>, !vhlo.tensor_v1<4x!vhlo.i32_v1>) -> !vhlo.tensor_v1<1x?x?x16x!vhlo.f32_v1>
  %0 = "stablehlo.dynamic_conv"(%arg0, %arg1, %arg2) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<4xi32>) -> tensor<1x?x?x16xf32>
  func.return %0 : tensor<1x?x?x16xf32>
}
// CHECK-LABEL: "default_dynamic_conv"

func.func @default_dynamic_gather(%arg0 : tensor<2x4x9xf32>, %arg1 : tensor<1x5x2xi32>, %arg2 : tensor<3xi32>) -> tensor<1x5x8xf32> {
  //      CHECK: "vhlo.dynamic_gather_v1"(%arg0, %arg1, %arg2) {
  // CHECK-SAME:   collapsed_slice_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   index_vector_dim = #vhlo.integer_v1<2 : i64>,
  // CHECK-SAME:   indices_are_sorted = #vhlo.bool_v1<false>,
  // CHECK-SAME:   offset_dims = #vhlo.tensor_v1<dense<2> : tensor<1xi64>>,
  // CHECK-SAME:   start_index_map = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<2x4x9x!vhlo.f32_v1>, !vhlo.tensor_v1<1x5x2x!vhlo.i32_v1>, !vhlo.tensor_v1<3x!vhlo.i32_v1>) -> !vhlo.tensor_v1<1x5x8x!vhlo.f32_v1>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %arg2) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >
  } : (tensor<2x4x9xf32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xf32>
  func.return %0 : tensor<1x5x8xf32>
}
// CHECK-LABEL: "default_dynamic_gather"

func.func @default_func(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK:      "vhlo.func_v1"() ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%arg0: !vhlo.tensor_v1<!vhlo.f32_v1>):
  // CHECK-NEXT:     "vhlo.return_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   arg_attrs = #vhlo.array_v1<[]>,
  // CHECK-SAME:   function_type = #vhlo.type_v1<!vhlo.func_v1<(!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>>>,
  // CHECK-SAME:   res_attrs = #vhlo.array_v1<[]>,
  // CHECK-SAME:   sym_name = #vhlo.string_v1<"default_func">,
  // CHECK-SAME:   sym_visibility = #vhlo.string_v1<"">
  // CHECK-SAME: } : () -> ()
  func.return %arg0 : tensor<f32>
}

func.func @dynamic_gather(%arg0 : tensor<2x4x9xf32>, %arg1 : tensor<1x5x2xi32>) -> tensor<1x5x1xf32> {
  //      CHECK: "vhlo.gather_v1"(%arg0, %arg1) {
  // CHECK-SAME:   collapsed_slice_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   index_vector_dim = #vhlo.integer_v1<2 : i64>,
  // CHECK-SAME:   indices_are_sorted = #vhlo.bool_v1<false>,
  // CHECK-SAME:   offset_dims = #vhlo.tensor_v1<dense<2> : tensor<1xi64>>,
  // CHECK-SAME:   slice_sizes = #vhlo.tensor_v1<dense<1> : tensor<3xi64>>,
  // CHECK-SAME:   start_index_map = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<2x4x9x!vhlo.f32_v1>, !vhlo.tensor_v1<1x5x2x!vhlo.i32_v1>) -> !vhlo.tensor_v1<1x5x1x!vhlo.f32_v1>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = dense<1> : tensor<3xi64>
  } : (tensor<2x4x9xf32>, tensor<1x5x2xi32>) -> tensor<1x5x1xf32>
  func.return %0 : tensor<1x5x1xf32>
}
// CHECK-LABEL: "dynamic_gather"

func.func @default_infeed(%arg0: !stablehlo.token) -> (tensor<f32>, !stablehlo.token) {
  //               CHECK: "vhlo.infeed_v1"(%arg0) {
  //          CHECK-SAME:   infeed_config = #vhlo.string_v1<"">,
  // CHECK-SAME{LITERAL}:   layout = #vhlo.array_v1<[]>
  //          CHECK-SAME: } : (!vhlo.token_v1) -> (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.token_v1)
  %0:2 = "stablehlo.infeed"(%arg0) : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !stablehlo.token
}
// CHECK-LABEL: "default_infeed"

func.func @default_outfeed(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  //      CHECK: "vhlo.outfeed_v1"(%arg0, %arg1) {
  // CHECK-SAME:   outfeed_config = #vhlo.string_v1<"">
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.token_v1) -> !vhlo.token_v1
  %0 = "stablehlo.outfeed"(%arg0, %arg1) : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}
// CHECK-LABEL: "default_outfeed"

func.func @default_recv(%arg0: !stablehlo.token) -> (tensor<f32>, !stablehlo.token) {
  //      CHECK: "vhlo.recv_v1"(%arg0) {
  // CHECK-SAME:   channel_id = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   channel_type = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   is_host_transfer = #vhlo.bool_v1<false>
  // CHECK-SAME: } : (!vhlo.token_v1) -> (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.token_v1)
  %0:2 = "stablehlo.recv"(%arg0) {
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !stablehlo.token
}
// CHECK-LABEL: "default_recv"

func.func @default_send(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  //      CHECK: "vhlo.send_v1"(%arg0, %arg1) {
  // CHECK-SAME:   channel_id = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   channel_type = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   is_host_transfer = #vhlo.bool_v1<false>
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.token_v1) -> !vhlo.token_v1
  %0 = "stablehlo.send"(%arg0, %arg1) {
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}
// CHECK-LABEL: "default_send"

func.func @default_reduce_scatter(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //               CHECK: "vhlo.reduce_scatter_v1"(%arg0) ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG2:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "vhlo.add_v1"(%[[ARG1]], %[[ARG2]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  //          CHECK-NEXT:     "vhlo.return_v1"(%[[VAL1]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  //          CHECK-NEXT: }) {
  //          CHECK-SAME:   channel_id = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME{LITERAL}:   replica_groups = #vhlo.tensor_v1<dense<[[0], [1]]> : tensor<2x1xi64>>,
  //          CHECK-SAME:   scatter_dimension = #vhlo.integer_v1<0 : i64>
  //          CHECK-SAME:   use_global_device_ids = #vhlo.bool_v1<false>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x!vhlo.f32_v1>
  %0 = "stablehlo.reduce_scatter"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    scatter_dimension = 0 : i64,
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "default_reduce_scatter"

func.func @default_reduce_window(%arg0: tensor<2x17x31x7xf32>, %arg1: tensor<f32>) -> tensor<2x16x30x7xf32> {
  //               CHECK: "vhlo.reduce_window_v1"(%arg0, %arg1) ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG2:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG3:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "vhlo.maximum_v1"(%[[ARG2]], %[[ARG3]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  //          CHECK-NEXT:     "vhlo.return_v1"(%[[VAL1]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  //          CHECK-NEXT: }) {
  //          CHECK-SAME:   base_dilations = #vhlo.tensor_v1<dense<1> : tensor<4xi64>>,
  // CHECK-SAME{LITERAL}:   padding = #vhlo.tensor_v1<dense<0> : tensor<4x2xi64>>,
  //          CHECK-SAME:   window_dilations = #vhlo.tensor_v1<dense<1> : tensor<4xi64>>,
  //          CHECK-SAME:   window_dimensions = #vhlo.tensor_v1<dense<[1, 2, 2, 1]> : tensor<4xi64>>,
  //          CHECK-SAME:   window_strides = #vhlo.tensor_v1<dense<1> : tensor<4xi64>>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<2x17x31x7x!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<2x16x30x7x!vhlo.f32_v1>
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = "stablehlo.maximum"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<2x17x31x7xf32>, tensor<f32>) -> tensor<2x16x30x7xf32>
  func.return %0 : tensor<2x16x30x7xf32>
}
// CHECK-LABEL: "default_reduce_window"

func.func @default_scatter(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>, %arg2: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
  //      CHECK: "vhlo.scatter_v1"(%arg0, %arg1, %arg2) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG3:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG4:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  // CHECK-NEXT:     %[[VAL1:.*]] = "vhlo.add_v1"(%[[ARG3]], %[[ARG4]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  // CHECK-NEXT:     "vhlo.return_v1"(%[[VAL1]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   index_vector_dim = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME:   indices_are_sorted = #vhlo.bool_v1<false>,
  // CHECK-SAME:   inserted_window_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   scatter_dims_to_operand_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   unique_indices = #vhlo.bool_v1<false>,
  // CHECK-SAME:   update_window_dims = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<200x100x300x!vhlo.f32_v1>, !vhlo.tensor_v1<10x2x!vhlo.i32_v1>, !vhlo.tensor_v1<10x300x!vhlo.f32_v1>) -> !vhlo.tensor_v1<200x100x300x!vhlo.f32_v1>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}
// CHECK-LABEL: "default_scatter"

func.func @default_select_and_scatter(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x23x23x64xf32>, %arg2: tensor<f32>) -> tensor<10x24x24x64xf32> {
  //      CHECK: "vhlo.select_and_scatter_v1"(%arg0, %arg1, %arg2) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG31:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG41:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  // CHECK-NEXT:     %[[VAL11:.*]] = "vhlo.compare_v1"(%[[ARG31]], %[[ARG41]]) {compare_type = #vhlo<comparison_type_v1 TOTALORDER>, comparison_direction = #vhlo<comparison_direction_v1 GE>} : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  // CHECK-NEXT:     "vhlo.return_v1"(%[[VAL11]]) : (!vhlo.tensor_v1<!vhlo.bool_v1>) -> ()
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG32:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG42:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  // CHECK-NEXT:     %[[VAL12:.*]] = "vhlo.add_v1"(%[[ARG32]], %[[ARG42]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  // CHECK-NEXT:     "vhlo.return_v1"(%[[VAL12]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   padding = #vhlo.tensor_v1<dense<0> : tensor<4x2xi64>>,
  // CHECK-SAME:   window_dimensions = #vhlo.tensor_v1<dense<[1, 2, 2, 1]> : tensor<4xi64>>,
  // CHECK-SAME:   window_strides = #vhlo.tensor_v1<dense<1> : tensor<4xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<10x24x24x64x!vhlo.f32_v1>, !vhlo.tensor_v1<10x23x23x64x!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<10x24x24x64x!vhlo.f32_v1>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "stablehlo.compare"(%arg3, %arg4) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<10x24x24x64xf32>, tensor<10x23x23x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  func.return %0 : tensor<10x24x24x64xf32>
}
// CHECK-LABEL: "default_select_and_scatter"

func.func @default_sort(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //      CHECK: "vhlo.sort_v1"(%arg0) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG2:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  // CHECK-NEXT:     %[[VAL1:.*]] = "vhlo.compare_v1"(%[[ARG1]], %[[ARG2]]) {compare_type = #vhlo<comparison_type_v1 FLOAT>, comparison_direction = #vhlo<comparison_direction_v1 GT>} : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  // CHECK-NEXT:     "vhlo.return_v1"(%[[VAL1]]) : (!vhlo.tensor_v1<!vhlo.bool_v1>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   dimension = #vhlo.integer_v1<-1 : i64>
  // CHECK-SAME:   is_stable = #vhlo.bool_v1<false>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x!vhlo.f32_v1>
  %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.compare"(%arg1, %arg2) {compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }) : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "default_sort"

// ============ OPS ============

func.func @op_abs(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.abs_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.abs"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_abs"

func.func @op_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_add"

func.func @op_after_all(%arg0: !stablehlo.token) -> !stablehlo.token {
  // CHECK: "vhlo.after_all_v1"(%arg0) : (!vhlo.token_v1) -> !vhlo.token_v1
  %0 = "stablehlo.after_all"(%arg0) : (!stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}
// CHECK-LABEL: "op_after_all"

func.func @op_all_gather(%arg0: tensor<16x8xf32>) -> tensor<16x16xf32> {
  //               CHECK: "vhlo.all_gather_v1"(%arg0) {
  //          CHECK-SAME:   all_gather_dim = #vhlo.integer_v1<1 : i64>
  //          CHECK-SAME:   channel_id = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME{LITERAL}:   replica_groups = #vhlo.tensor_v1<dense<[[0], [1]]> : tensor<2x1xi64>>,
  //          CHECK-SAME:   use_global_device_ids = #vhlo.bool_v1<true>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x16x!vhlo.f32_v1>
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    use_global_device_ids
  } : (tensor<16x8xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "op_all_gather"

func.func @op_all_reduce(%arg0: tensor<f32>) -> tensor<f32> {
  //               CHECK: "vhlo.all_reduce_v1"(%arg0) ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG2:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "vhlo.add_v1"(%[[ARG1]], %[[ARG2]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  //          CHECK-NEXT:     "vhlo.return_v1"(%[[VAL1]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  //          CHECK-NEXT: }) {
  //          CHECK-SAME:   channel_id = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME{LITERAL}:   replica_groups = #vhlo.tensor_v1<dense<[[0], [1]]> : tensor<2x1xi64>>,
  //          CHECK-SAME:   use_global_device_ids = #vhlo.bool_v1<true>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    use_global_device_ids
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_all_reduce"

func.func @op_all_to_all(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
  //               CHECK: "vhlo.all_to_all_v1"(%arg0) {
  //          CHECK-SAME:   channel_id = #vhlo.integer_v1<1 : i64>,
  //          CHECK-SAME:   concat_dimension = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME{LITERAL}:   replica_groups = #vhlo.tensor_v1<dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>>,
  //          CHECK-SAME:   split_count = #vhlo.integer_v1<4 : i64>
  //          CHECK-SAME:   split_dimension = #vhlo.integer_v1<1 : i64>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<4x16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x4x!vhlo.f32_v1>
  %0 = "stablehlo.all_to_all"(%arg0) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}
// CHECK-LABEL: "op_all_to_all"

func.func @op_and(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "vhlo.and_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.bool_v1>, !vhlo.tensor_v1<!vhlo.bool_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  %0 = "stablehlo.and"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_and"

func.func @op_atan2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.atan2_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.atan2"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_atan2"

func.func @op_batch_norm_grad(%arg0: tensor<16x16x16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16x16x16x16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>) {
  //      CHECK: "vhlo.batch_norm_grad_v1"(%arg0, %arg1, %arg2, %arg3, %arg4) {
  // CHECK-SAME:   epsilon = #vhlo.float_v1<1.000000e-03 : !vhlo.f32_v1>,
  // CHECK-SAME:   feature_index = #vhlo.integer_v1<0 : i64>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x16x16x16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x16x16x16x!vhlo.f32_v1>) -> (!vhlo.tensor_v1<16x16x16x16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>)
  %0:3 = "stablehlo.batch_norm_grad"(%arg0, %arg1, %arg2, %arg3, %arg4) {
    epsilon = 0.001 : f32,
    feature_index = 0 : i64
  } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x16x16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>
}
// CHECK-LABEL: "op_batch_norm_grad"

func.func @op_batch_norm_inference(%arg0: tensor<16x16x16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x16x16x16xf32> {
  //      CHECK: "vhlo.batch_norm_inference_v1"(%arg0, %arg1, %arg2, %arg3, %arg4) {
  // CHECK-SAME:   epsilon = #vhlo.float_v1<1.000000e-03 : !vhlo.f32_v1>,
  // CHECK-SAME:   feature_index = #vhlo.integer_v1<0 : i64>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x16x16x16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x16x16x16x!vhlo.f32_v1>
  %0 = "stablehlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {
    epsilon = 0.001 : f32,
    feature_index = 0 : i64
  } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x16x16x16xf32>
  func.return %0 : tensor<16x16x16x16xf32>
}
// CHECK-LABEL: "op_batch_norm_inference"

func.func @op_batch_norm_training(%arg0: tensor<16x16x16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>) {
  //      CHECK: "vhlo.batch_norm_training_v1"(%arg0, %arg1, %arg2) {
  // CHECK-SAME:   epsilon = #vhlo.float_v1<1.000000e-03 : !vhlo.f32_v1>,
  // CHECK-SAME:   feature_index = #vhlo.integer_v1<0 : i64>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x16x16x16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>) -> (!vhlo.tensor_v1<16x16x16x16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x!vhlo.f32_v1>)
  %0:3 = "stablehlo.batch_norm_training"(%arg0, %arg1, %arg2) {
    epsilon = 0.001 : f32,
    feature_index = 0 : i64
  } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>
}
// CHECK-LABEL: "op_batch_norm_training"

func.func @op_bitcast_convert(%arg0: tensor<i32>) -> tensor<f32> {
  // CHECK: "vhlo.bitcast_convert_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.i32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.bitcast_convert"(%arg0) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_bitcast_convert"

func.func @op_broadcast_in_dim(%arg0: tensor<16xf32>) -> tensor<16x16xf32> {
  //      CHECK: "vhlo.broadcast_in_dim_v1"(%arg0) {
  // CHECK-SAME:   broadcast_dimensions = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x16x!vhlo.f32_v1>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<1> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "op_broadcast_in_dim"

func.func @op_broadcast(%arg0: tensor<16xf32>) -> tensor<16x16xf32> {
  //      CHECK: "vhlo.broadcast_v1"(%arg0) {
  // CHECK-SAME:   broadcast_sizes = #vhlo.tensor_v1<dense<16> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x16x!vhlo.f32_v1>
  %0 = "stablehlo.broadcast"(%arg0) {
    broadcast_sizes = dense<16> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "op_broadcast"

func.func @op_case(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
  //      CHECK: "vhlo.case_v1"(%arg0) ({
  // CHECK-NEXT:   "vhlo.return_v1"(%arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  // CHECK-NEXT: }) : (!vhlo.tensor_v1<!vhlo.i32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.case"(%arg0) ({
    "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_case"

func.func @op_cbrt(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.cbrt_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.cbrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_cbrt"

func.func @op_ceil(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.ceil_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.ceil"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_ceil"

func.func @op_cholesky(%arg0: tensor<1x16x16xf32>) -> tensor<1x16x16xf32> {
  //      CHECK: "vhlo.cholesky_v1"(%arg0) {
  // CHECK-SAME:   lower = #vhlo.bool_v1<true>
  // CHECK-SAME: } : (!vhlo.tensor_v1<1x16x16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<1x16x16x!vhlo.f32_v1>
  %0 = "stablehlo.cholesky"(%arg0) {
    lower = true
  } : (tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
  func.return %0 : tensor<1x16x16xf32>
}
// CHECK-LABEL: "op_cholesky"

func.func @op_clamp(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.clamp_v1"(%arg0, %arg1, %arg2) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.clamp"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_clamp"

func.func @op_count_leading_zeros(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: "vhlo.count_leading_zeros_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.i32_v1>) -> !vhlo.tensor_v1<!vhlo.i32_v1>
  %0 = "stablehlo.count_leading_zeros"(%arg0) : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_count_leading_zeros"

func.func @op_collective_permute(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  //               CHECK: "vhlo.collective_permute_v1"(%arg0) {
  //          CHECK-SAME:   channel_id = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME{LITERAL}:   source_target_pairs = #vhlo.tensor_v1<dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x8x!vhlo.f32_v1>
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}
// CHECK-LABEL: "op_collective_permute"

func.func @op_compare(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  //      CHECK: "vhlo.compare_v1"(%arg0, %arg1) {
  // CHECK-SAME:   compare_type = #vhlo<comparison_type_v1 TOTALORDER>,
  // CHECK-SAME:   comparison_direction = #vhlo<comparison_direction_v1 EQ>
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    compare_type = #stablehlo<comparison_type TOTALORDER>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_compare"

func.func @op_complex(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<complex<f32>> {
  // CHECK: "vhlo.complex_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.complex_v1<!vhlo.f32_v1>>
  %0 = "stablehlo.complex"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
  func.return %0 : tensor<complex<f32>>
}
// CHECK-LABEL: "op_complex"

func.func @op_compute_reshape_shape(%arg0: index, %arg1: tensor<1xindex>) -> tensor<1xindex> {
  // CHECK: "vhlo.compute_reshape_shape_v1"(%arg0, %arg1) : (!vhlo.index_v1, !vhlo.tensor_v1<1x!vhlo.index_v1>) -> !vhlo.tensor_v1<1x!vhlo.index_v1>
  %0 = "stablehlo.compute_reshape_shape"(%arg0, %arg1) : (index, tensor<1xindex>) -> tensor<1xindex>
  func.return %0 : tensor<1xindex>
}
// CHECK-LABEL: "op_compute_reshape_shape"

func.func @op_concatenate(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<16xf32> {
  //      CHECK: "vhlo.concatenate_v1"(%arg0, %arg1) {
  // CHECK-SAME:   dimension = #vhlo.integer_v1<0 : i64>
  // CHECK-SAME: } : (!vhlo.tensor_v1<8x!vhlo.f32_v1>, !vhlo.tensor_v1<8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x!vhlo.f32_v1>
  %0 = "stablehlo.concatenate"(%arg0, %arg1) {
    dimension = 0 : i64
  } : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_concatenate"

func.func @op_constant(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "vhlo.constant_v1"() {
  // CHECK-SAME:   value = #vhlo.tensor_v1<dense<0.000000e+00> : tensor<f32>>
  // CHECK-SAME: } : () -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.constant"() {
    value = dense<0.0> : tensor<f32>
  } : () -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_constant"

func.func @op_convert(%arg0: tensor<i32>) -> tensor<f32> {
  // CHECK: "vhlo.convert_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.i32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.convert"(%arg0) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_convert"

func.func @op_convolution(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>) -> tensor<1x7x7x16xf32> {
  //      CHECK: "vhlo.convolution_v1"(%arg0, %arg1) {
  // CHECK-SAME:   batch_group_count = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME:   feature_group_count = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME:   input_batch_dimension = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   input_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   input_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
  // CHECK-SAME:   kernel_input_feature_dimension = #vhlo.integer_v1<2 : i64>,
  // CHECK-SAME:   kernel_output_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   kernel_spatial_dimensions = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   lhs_dilation = #vhlo.tensor_v1<dense<2> : tensor<2xi64>>,
  // CHECK-SAME:   output_batch_dimension = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   output_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   output_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
  // CHECK-SAME:   padding = #vhlo.tensor_v1<dense<1> : tensor<2x2xi64>>,
  // CHECK-SAME:   precision_config = #vhlo.array_v1<[#vhlo<precision_v1 HIGHEST>, #vhlo<precision_v1 HIGHEST>]>,
  // CHECK-SAME:   rhs_dilation = #vhlo.tensor_v1<dense<2> : tensor<2xi64>>,
  // CHECK-SAME:   window_reversal = #vhlo.tensor_v1<dense<true> : tensor<2xi1>>,
  // CHECK-SAME:   window_strides = #vhlo.tensor_v1<dense<2> : tensor<2xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<1x8x8x207x!vhlo.f32_v1>, !vhlo.tensor_v1<3x3x207x16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<1x7x7x16x!vhlo.f32_v1>
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    window_strides = dense<2> : tensor<2xi64>,
    padding = dense<1> : tensor<2x2xi64>,
    lhs_dilation = dense<2> : tensor<2xi64>,
    rhs_dilation = dense<2> : tensor<2xi64>,
    window_reversal = dense<true> : tensor<2xi1>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x7x7x16xf32>
  func.return %0 : tensor<1x7x7x16xf32>
}
// CHECK-LABEL: "op_convolution"

func.func @op_cosine(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.cosine_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.cosine"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_cosine"

func.func @op_create_token() -> !stablehlo.token {
  // CHECK: "vhlo.create_token_v1"() : () -> !vhlo.token_v1
  %0 = "stablehlo.create_token"() : () -> !stablehlo.token
  func.return %0 : !stablehlo.token
}
// CHECK-LABEL: "op_create_token"

func.func @op_cross_replica_sum(%arg0: tensor<f32>) -> tensor<f32> {
  //               CHECK: "vhlo.cross-replica-sum_v1"(%arg0) {
  // CHECK-SAME{LITERAL}:   replica_groups = #vhlo.tensor_v1<dense<[[0], [1]]> : tensor<2x1xi64>>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.cross-replica-sum"(%arg0) {
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_cross_replica_sum"

func.func @op_cstr_reshapable(%arg0: index, %arg1: tensor<1xindex>) -> !shape.witness {
  // CHECK: "vhlo.cstr_reshapable_v1"(%arg0, %arg1) : (!vhlo.index_v1, !vhlo.tensor_v1<1x!vhlo.index_v1>) -> !vhlo.witness_v1
  %0 = "stablehlo.cstr_reshapable"(%arg0, %arg1) : (index, tensor<1xindex>) -> !shape.witness
  func.return %0 : !shape.witness
}
// CHECK-LABEL: "op_cstr_reshapable"

func.func @op_custom_call(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "vhlo.custom_call_v1"(%arg0) {
  // CHECK-SAME:   api_version = #vhlo<api_version_v1 API_VERSION_STATUS_RETURNING>,
  // CHECK-SAME:   backend_config = #vhlo.string_v1<"\08\03\1A\02">,
  // CHECK-SAME:   call_target_name = #vhlo.string_v1<"foo">,
  // CHECK-SAME:   called_computations = #vhlo.array_v1<[#vhlo.string_v1<"foo">]>,
  // CHECK-SAME:   has_side_effect = #vhlo.bool_v1<true>,
  // CHECK-SAME:   operand_layouts = #vhlo.array_v1<[#vhlo.tensor_v1<dense<> : tensor<0xindex>>]>,
  // CHECK-SAME:   output_operand_aliases = #vhlo.array_v1<[
  // CHECK-SAME:     #vhlo.output_operand_alias_v1<
  // CHECK-SAME:       outputTupleIndices = [],
  // CHECK-SAME:       operandIndex = 0,
  // CHECK-SAME:       operandTupleIndices = []>]>
  // CHECK-SAME:   result_layouts = #vhlo.array_v1<[#vhlo.tensor_v1<dense<> : tensor<0xindex>>]>
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    has_side_effect = true,
    backend_config = "\08\03\1A\02",
    api_version = 2 : i32,
    called_computations = [@foo],
    operand_layouts = [dense<> : tensor<0xindex>],
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = []>],
    result_layouts = [dense<> : tensor<0xindex>]
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_custom_call"

func.func @op_divide(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.divide_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.divide"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_divide"

func.func @op_dot_general(%arg0: tensor<8x8x16xf32>, %arg1: tensor<8x16x8xf32>) -> tensor<8x8x8xf32> {
  //      CHECK: "vhlo.dot_general_v1"(%arg0, %arg1) {
  // CHECK-SAME:   lhs_batching_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
  // CHECK-SAME:   lhs_contracting_dimensions = #vhlo.tensor_v1<dense<2> : tensor<1xi64>>,
  // CHECK-SAME:   precision_config = #vhlo.array_v1<[#vhlo<precision_v1 HIGHEST>, #vhlo<precision_v1 HIGHEST>]>,
  // CHECK-SAME:   rhs_batching_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
  // CHECK-SAME:   rhs_contracting_dimensions = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<8x8x16x!vhlo.f32_v1>, !vhlo.tensor_v1<8x16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<8x8x8x!vhlo.f32_v1>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]
  } : (tensor<8x8x16xf32>, tensor<8x16x8xf32>) -> tensor<8x8x8xf32>
  func.return %0 : tensor<8x8x8xf32>
}
// CHECK-LABEL: "op_dot_general"

func.func @op_dot(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  //      CHECK: "vhlo.dot_v1"(%arg0, %arg1) {
  // CHECK-SAME:   precision_config = #vhlo.array_v1<[#vhlo<precision_v1 HIGHEST>, #vhlo<precision_v1 HIGHEST>]>
  // CHECK-SAME: } : (!vhlo.tensor_v1<8x16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<8x8x!vhlo.f32_v1>
  %0 = "stablehlo.dot"(%arg0, %arg1) {
    precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: "op_dot"

func.func @op_dynamic_broadcast_in_dim(%arg0: tensor<?x?xf32>, %arg1: tensor<2xindex>) -> tensor<?x?xf32> {
  //      CHECK: "vhlo.dynamic_broadcast_in_dim_v1"(%arg0, %arg1) {
  // CHECK-SAME:   broadcast_dimensions = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   known_expanding_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
  // CHECK-SAME:   known_nonexpanding_dimensions = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<?x?x!vhlo.f32_v1>, !vhlo.tensor_v1<2x!vhlo.index_v1>) -> !vhlo.tensor_v1<?x?x!vhlo.f32_v1>
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>,
    known_expanding_dimensions = dense<0> : tensor<1xi64>,
    known_nonexpanding_dimensions = dense<1> : tensor<1xi64>
  } : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: "op_dynamic_broadcast_in_dim"

func.func @op_dynamic_conv(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>, %arg2: tensor<4xi32>) -> tensor<1x?x?x16xf32> {
  //      CHECK: "vhlo.dynamic_conv_v1"(%arg0, %arg1, %arg2) {
  // CHECK-SAME:   batch_group_count = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME:   feature_group_count = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME:   input_batch_dimension = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   input_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   input_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
  // CHECK-SAME:   kernel_input_feature_dimension = #vhlo.integer_v1<2 : i64>,
  // CHECK-SAME:   kernel_output_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   kernel_spatial_dimensions = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   lhs_dilation = #vhlo.tensor_v1<dense<2> : tensor<2xi64>>,
  // CHECK-SAME:   output_batch_dimension = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   output_feature_dimension = #vhlo.integer_v1<3 : i64>,
  // CHECK-SAME:   output_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
  // CHECK-SAME:   padding = #vhlo.tensor_v1<dense<1> : tensor<2x2xi64>>,
  // CHECK-SAME:   precision_config = #vhlo.array_v1<[#vhlo<precision_v1 HIGHEST>, #vhlo<precision_v1 HIGHEST>]>,
  // CHECK-SAME:   rhs_dilation = #vhlo.tensor_v1<dense<2> : tensor<2xi64>>,
  // CHECK-SAME:   window_reversal = #vhlo.tensor_v1<dense<true> : tensor<2xi1>>,
  // CHECK-SAME:   window_strides = #vhlo.tensor_v1<dense<2> : tensor<2xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<1x8x8x207x!vhlo.f32_v1>, !vhlo.tensor_v1<3x3x207x16x!vhlo.f32_v1>, !vhlo.tensor_v1<4x!vhlo.i32_v1>) -> !vhlo.tensor_v1<1x?x?x16x!vhlo.f32_v1>
  %0 = "stablehlo.dynamic_conv"(%arg0, %arg1, %arg2) {
    window_strides = dense<2> : tensor<2xi64>,
    padding = dense<1> : tensor<2x2xi64>,
    lhs_dilation = dense<2> : tensor<2xi64>,
    rhs_dilation = dense<2> : tensor<2xi64>,
    window_reversal = dense<true> : tensor<2xi1>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<4xi32>) -> tensor<1x?x?x16xf32>
  func.return %0 : tensor<1x?x?x16xf32>
}
// CHECK-LABEL: "op_dynamic_conv"

func.func @op_dynamic_gather(%arg0 : tensor<2x4x9xf32>, %arg1 : tensor<1x5x2xi32>, %arg2 : tensor<3xi32>) -> tensor<1x5x8xf32> {
  //      CHECK: "vhlo.dynamic_gather_v1"(%arg0, %arg1, %arg2) {
  // CHECK-SAME:   collapsed_slice_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   index_vector_dim = #vhlo.integer_v1<2 : i64>,
  // CHECK-SAME:   indices_are_sorted = #vhlo.bool_v1<true>,
  // CHECK-SAME:   offset_dims = #vhlo.tensor_v1<dense<2> : tensor<1xi64>>,
  // CHECK-SAME:   start_index_map = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<2x4x9x!vhlo.f32_v1>, !vhlo.tensor_v1<1x5x2x!vhlo.i32_v1>, !vhlo.tensor_v1<3x!vhlo.i32_v1>) -> !vhlo.tensor_v1<1x5x8x!vhlo.f32_v1>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %arg2) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    indices_are_sorted = true
  } : (tensor<2x4x9xf32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xf32>
  func.return %0 : tensor<1x5x8xf32>
}
// CHECK-LABEL: "op_dynamic_gather"

func.func @op_dynamic_iota(%arg0: tensor<1xindex>) -> tensor<?xf32> {
  //      CHECK: "vhlo.dynamic_iota_v1"(%arg0) {
  // CHECK-SAME:   iota_dimension = #vhlo.integer_v1<0 : i64>
  // CHECK-SAME: } : (!vhlo.tensor_v1<1x!vhlo.index_v1>) -> !vhlo.tensor_v1<?x!vhlo.f32_v1>
  %0 = "stablehlo.dynamic_iota"(%arg0) {
    iota_dimension = 0 : i64
  } : (tensor<1xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: "op_dynamic_iota"

func.func @op_dynamic_pad(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>, %arg4: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK: "vhlo.dynamic_pad_v1"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!vhlo.tensor_v1<?x!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<1x!vhlo.index_v1>, !vhlo.tensor_v1<1x!vhlo.index_v1>, !vhlo.tensor_v1<1x!vhlo.index_v1>) -> !vhlo.tensor_v1<?x!vhlo.f32_v1>
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<?xf32>, tensor<f32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: "op_dynamic_pad"

func.func @op_dynamic_reshape(%arg0: tensor<16xf32>, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  // CHECK: "vhlo.dynamic_reshape_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<?x!vhlo.index_v1>) -> !vhlo.tensor_v1<?x?x!vhlo.f32_v1>
  %0 = "stablehlo.dynamic_reshape"(%arg0, %arg1) : (tensor<16xf32>, tensor<?xindex>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: "op_dynamic_reshape"

func.func @op_dynamic_slice(%arg0: tensor<16xf32>, %arg1: tensor<i64>) -> tensor<4xf32> {
  //      CHECK: "vhlo.dynamic_slice_v1"(%arg0, %arg1) {
  // CHECK-SAME:   slice_sizes = #vhlo.tensor_v1<dense<4> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.i64_v1>) -> !vhlo.tensor_v1<4x!vhlo.f32_v1>
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1) {
    slice_sizes = dense<4> : tensor<1xi64>
  } : (tensor<16xf32>, tensor<i64>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
// CHECK-LABEL: "op_dynamic_slice"

func.func @op_dynamic_update_slice(%arg0: tensor<16xf32>, %arg1: tensor<4xf32>, %arg2: tensor<i64>) -> tensor<16xf32> {
  // CHECK: "vhlo.dynamic_update_slice_v1"(%arg0, %arg1, %arg2) : (!vhlo.tensor_v1<16x!vhlo.f32_v1>, !vhlo.tensor_v1<4x!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.i64_v1>) -> !vhlo.tensor_v1<16x!vhlo.f32_v1>
  %0 = "stablehlo.dynamic_update_slice"(%arg0, %arg1, %arg2) : (tensor<16xf32>, tensor<4xf32>, tensor<i64>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_dynamic_update_slice"

func.func @op_einsum(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  //      CHECK: "vhlo.einsum_v1"(%arg0, %arg1) {
  // CHECK-SAME:   einsum_config = #vhlo.string_v1<"ab,bc->ac">
  // CHECK-SAME: } : (!vhlo.tensor_v1<8x16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<8x8x!vhlo.f32_v1>
  %0 = "stablehlo.einsum"(%arg0, %arg1) {
    einsum_config = "ab,bc->ac"
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: "op_einsum"

func.func @op_exponential_minus_one(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.exponential_minus_one_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.exponential_minus_one"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_exponential_minus_one"

func.func @op_exponential(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.exponential_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.exponential"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_exponential"

func.func @op_fft(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  //      CHECK: "vhlo.fft_v1"(%arg0) {
  // CHECK-SAME:   fft_length = #vhlo.tensor_v1<dense<16> : tensor<1xi64>>,
  // CHECK-SAME:   fft_type = #vhlo<fft_type_v1 FFT>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x!vhlo.complex_v1<!vhlo.f32_v1>>) -> !vhlo.tensor_v1<16x!vhlo.complex_v1<!vhlo.f32_v1>>
  %0 = "stablehlo.fft"(%arg0) {
    fft_type = #stablehlo<fft_type FFT>,
    fft_length = dense<16> : tensor<1xi64>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}
// CHECK-LABEL: "op_fft"

func.func @op_floor(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.floor_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_floor"

func.func private @op_func(%arg0: tensor<f32> {stablehlo.arg = "0"}) -> (tensor<f32> {stablehlo.result = "0"}) {
  // CHECK:      "vhlo.func_v1"() ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%arg0: !vhlo.tensor_v1<!vhlo.f32_v1>):
  // CHECK-NEXT:     "vhlo.return_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   arg_attrs = #vhlo.array_v1<[#vhlo.dict_v1<{#vhlo.string_v1<"stablehlo.arg"> = #vhlo.string_v1<"0">}>]>,
  // CHECK-SAME:   function_type = #vhlo.type_v1<!vhlo.func_v1<(!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>>>,
  // CHECK-SAME:   res_attrs = #vhlo.array_v1<[#vhlo.dict_v1<{#vhlo.string_v1<"stablehlo.result"> = #vhlo.string_v1<"0">}>]>,
  // CHECK-SAME:   sym_name = #vhlo.string_v1<"op_func">,
  // CHECK-SAME:   sym_visibility = #vhlo.string_v1<"private">
  // CHECK-SAME: } : () -> ()
  func.return %arg0 : tensor<f32>
}

func.func @op_gather(%arg0 : tensor<2x4x9xf32>, %arg1 : tensor<1x5x2xi32>) -> tensor<1x5x1xf32> {
  //      CHECK: "vhlo.gather_v1"(%arg0, %arg1) {
  // CHECK-SAME:   collapsed_slice_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   index_vector_dim = #vhlo.integer_v1<2 : i64>,
  // CHECK-SAME:   indices_are_sorted = #vhlo.bool_v1<true>,
  // CHECK-SAME:   offset_dims = #vhlo.tensor_v1<dense<2> : tensor<1xi64>>,
  // CHECK-SAME:   slice_sizes = #vhlo.tensor_v1<dense<1> : tensor<3xi64>>,
  // CHECK-SAME:   start_index_map = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<2x4x9x!vhlo.f32_v1>, !vhlo.tensor_v1<1x5x2x!vhlo.i32_v1>) -> !vhlo.tensor_v1<1x5x1x!vhlo.f32_v1>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = dense<1> : tensor<3xi64>,
    indices_are_sorted = true
  } : (tensor<2x4x9xf32>, tensor<1x5x2xi32>) -> tensor<1x5x1xf32>
  func.return %0 : tensor<1x5x1xf32>
}
// CHECK-LABEL: "op_gather"

func.func @op_get_dimension_size(%arg0: tensor<?xf32>) -> tensor<i32> {
  //      CHECK: "vhlo.get_dimension_size_v1"(%arg0) {
  // CHECK-SAME:   dimension = #vhlo.integer_v1<0 : i64>
  // CHECK-SAME: } : (!vhlo.tensor_v1<?x!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.i32_v1>
  %0 = "stablehlo.get_dimension_size"(%arg0) {
    dimension = 0 : i64
  } : (tensor<?xf32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_get_dimension_size"

func.func @op_get_tuple_element(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  //      CHECK: "vhlo.get_tuple_element_v1"(%arg0) {
  // CHECK-SAME:   index = #vhlo.integer_v1<0 : i32>
  // CHECK-SAME: } : (!vhlo.tuple_v1<!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.i32_v1>>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.get_tuple_element"(%arg0) {
    index = 0 : i32
  } : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_get_tuple_element"

func.func @op_if(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  //      CHECK: "vhlo.if_v1"(%arg0) ({
  // CHECK-NEXT:   "vhlo.return_v1"(%arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   "vhlo.return_v1"(%arg2) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  // CHECK-NEXT: }) : (!vhlo.tensor_v1<!vhlo.bool_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.if"(%arg0) ({
    "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  }, {
    "stablehlo.return"(%arg2) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_if"

func.func @op_imag(%arg0: tensor<complex<f32>>) -> tensor<f32> {
  // CHECK: "vhlo.imag_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.complex_v1<!vhlo.f32_v1>>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.imag"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_imag"

func.func @op_infeed(%arg0: !stablehlo.token) -> (tensor<f32>, !stablehlo.token) {
  //               CHECK: "vhlo.infeed_v1"(%arg0) {
  //          CHECK-SAME:   infeed_config = #vhlo.string_v1<"foo">,
  // CHECK-SAME{LITERAL}:   layout = #vhlo.array_v1<[#vhlo.array_v1<[]>]>
  //          CHECK-SAME: } : (!vhlo.token_v1) -> (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.token_v1)
  %0:2 = "stablehlo.infeed"(%arg0) {
    infeed_config = "foo",
    layout = [[]]
  } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !stablehlo.token
}
// CHECK-LABEL: "op_infeed"

func.func @op_iota() -> tensor<16xf32> {
  //      CHECK: "vhlo.iota_v1"() {
  // CHECK-SAME:   iota_dimension = #vhlo.integer_v1<0 : i64>
  // CHECK-SAME: } : () -> !vhlo.tensor_v1<16x!vhlo.f32_v1>
  %0 = "stablehlo.iota"() {
    iota_dimension = 0 : i64
  } : () -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_iota"

func.func @op_is_finite(%arg0: tensor<f32>) -> tensor<i1> {
  // CHECK: "vhlo.is_finite_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  %0 = "stablehlo.is_finite"(%arg0) : (tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_is_finite"

func.func @op_log(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.log_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_log"

func.func @op_log_plus_one(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.log_plus_one_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_log_plus_one"

func.func @op_logistic(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.logistic_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.logistic"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_logistic"

func.func @op_map(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //      CHECK: "vhlo.map_v1"(%arg0) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  // CHECK-NEXT:     %[[VAL1:.*]] = "vhlo.abs_v1"(%[[ARG1]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  // CHECK-NEXT:     "vhlo.return_v1"(%[[VAL1]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x!vhlo.f32_v1>
  %0 = "stablehlo.map"(%arg0) ({
    ^bb0(%arg1: tensor<f32>):
      %1 = "stablehlo.abs"(%arg1) : (tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    dimensions = dense<0> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_map"

func.func @op_maximum(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.maximum_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_maximum"

func.func @op_minimum(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.minimum_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_minimum"

func.func @op_multiply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.multiply_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_multiply"

func.func @op_negate(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.negate_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_negate"

func.func @op_not(%arg0: tensor<i1>) -> tensor<i1> {
  // CHECK: "vhlo.not_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.bool_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  %0 = "stablehlo.not"(%arg0) : (tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_not"

func.func @op_optimization_barrier(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.optimization_barrier_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.optimization_barrier"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_optimization_barrier"

func.func @op_or(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "vhlo.or_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.bool_v1>, !vhlo.tensor_v1<!vhlo.bool_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  %0 = "stablehlo.or"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_or"

func.func @op_outfeed(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  //      CHECK: "vhlo.outfeed_v1"(%arg0, %arg1) {
  // CHECK-SAME:   outfeed_config = #vhlo.string_v1<"foo">
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.token_v1) -> !vhlo.token_v1
  %0 = "stablehlo.outfeed"(%arg0, %arg1) {
    outfeed_config = "foo"
  } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}
// CHECK-LABEL: "op_outfeed"

func.func @op_pad(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
  //      CHECK: "vhlo.pad_v1"(%arg0, %arg1) {
  // CHECK-SAME:   edge_padding_high = #vhlo.tensor_v1<dense<4> : tensor<1xi64>>,
  // CHECK-SAME:   edge_padding_low = #vhlo.tensor_v1<dense<4> : tensor<1xi64>>,
  // CHECK-SAME:   interior_padding = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<8x!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x!vhlo.f32_v1>
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<4> : tensor<1xi64>,
    edge_padding_low = dense<4> : tensor<1xi64>,
    interior_padding = dense<0> : tensor<1xi64>
  } : (tensor<8xf32>, tensor<f32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_pad"

func.func @op_popcnt(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: "vhlo.popcnt_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.i32_v1>) -> !vhlo.tensor_v1<!vhlo.i32_v1>
  %0 = "stablehlo.popcnt"(%arg0) : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_popcnt"

func.func @op_power(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.power_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.power"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_power"

func.func @op_real_dynamic_slice(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK: "vhlo.real_dynamic_slice_v1"(%arg0, %arg1, %arg2, %arg3) : (!vhlo.tensor_v1<?x!vhlo.f32_v1>, !vhlo.tensor_v1<1x!vhlo.index_v1>, !vhlo.tensor_v1<1x!vhlo.index_v1>, !vhlo.tensor_v1<1x!vhlo.index_v1>) -> !vhlo.tensor_v1<?x!vhlo.f32_v1>
  %0 = "stablehlo.real_dynamic_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: "op_real_dynamic_slice"

func.func @op_real(%arg0: tensor<complex<f32>>) -> tensor<f32> {
  // CHECK: "vhlo.real_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.complex_v1<!vhlo.f32_v1>>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.real"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_real"

func.func @op_recv(%arg0: !stablehlo.token) -> (tensor<f32>, !stablehlo.token) {
  //      CHECK: "vhlo.recv_v1"(%arg0) {
  // CHECK-SAME:   channel_id = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   channel_type = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   is_host_transfer = #vhlo.bool_v1<true>
  // CHECK-SAME: } : (!vhlo.token_v1) -> (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.token_v1)
  %0:2 = "stablehlo.recv"(%arg0) {
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
    is_host_transfer = true
  } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !stablehlo.token
}
// CHECK-LABEL: "op_recv"

func.func @op_reduce(%arg0: tensor<16xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    dimensions = dense<0> : tensor<1xi64>
  } : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_reduce"

func.func @op_reduce_precision(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "vhlo.reduce_precision_v1"(%arg0) {
  // CHECK-SAME:   exponent_bits = #vhlo.integer_v1<8 : i32>
  // CHECK-SAME:   mantissa_bits = #vhlo.integer_v1<10 : i32>
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.reduce_precision"(%arg0) {
    exponent_bits = 8 : i32,
    mantissa_bits = 10 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_reduce_precision"

func.func @op_reduce_scatter(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //               CHECK: "vhlo.reduce_scatter_v1"(%arg0) ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG2:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "vhlo.add_v1"(%[[ARG1]], %[[ARG2]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  //          CHECK-NEXT:     "vhlo.return_v1"(%[[VAL1]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  //          CHECK-NEXT: }) {
  //          CHECK-SAME:   channel_id = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME{LITERAL}:   replica_groups = #vhlo.tensor_v1<dense<[[0], [1]]> : tensor<2x1xi64>>,
  //          CHECK-SAME:   scatter_dimension = #vhlo.integer_v1<0 : i64>
  //          CHECK-SAME:   use_global_device_ids = #vhlo.bool_v1<true>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x!vhlo.f32_v1>
  %0 = "stablehlo.reduce_scatter"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    scatter_dimension = 0 : i64,
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    use_global_device_ids
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_reduce_scatter"

func.func @op_reduce_window(%arg0: tensor<2x17x31x7xf32>, %arg1: tensor<f32>) -> tensor<2x9x16x7xf32> {
  //               CHECK: "vhlo.reduce_window_v1"(%arg0, %arg1) ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG2:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG3:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "vhlo.maximum_v1"(%[[ARG2]], %[[ARG3]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  //          CHECK-NEXT:     "vhlo.return_v1"(%[[VAL1]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  //          CHECK-NEXT: }) {
  //          CHECK-SAME:   base_dilations = #vhlo.tensor_v1<dense<[1, 2, 2, 1]> : tensor<4xi64>>,
  // CHECK-SAME{LITERAL}:   padding = #vhlo.tensor_v1<dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>>,
  //          CHECK-SAME:   window_dilations = #vhlo.tensor_v1<dense<[1, 2, 2, 1]> : tensor<4xi64>>,
  //          CHECK-SAME:   window_dimensions = #vhlo.tensor_v1<dense<[1, 2, 2, 1]> : tensor<4xi64>>,
  //          CHECK-SAME:   window_strides = #vhlo.tensor_v1<dense<[1, 4, 4, 1]> : tensor<4xi64>>
  //          CHECK-SAME: } : (!vhlo.tensor_v1<2x17x31x7x!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<2x9x16x7x!vhlo.f32_v1>
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = "stablehlo.maximum"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>,
    base_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>
  } : (tensor<2x17x31x7xf32>, tensor<f32>) -> tensor<2x9x16x7xf32>
  func.return %0 : tensor<2x9x16x7xf32>
}
// CHECK-LABEL: "op_reduce_window"

func.func @op_remainder(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.remainder_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.remainder"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_remainder"

func.func @op_replica_id() -> tensor<ui32> {
  // CHECK: "vhlo.replica_id_v1"() : () -> !vhlo.tensor_v1<!vhlo.ui32_v1>
  %0 = "stablehlo.replica_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-LABEL: "op_replica_id"

func.func @op_partition_id() -> tensor<ui32> {
  // CHECK: "vhlo.partition_id_v1"() : () -> !vhlo.tensor_v1<!vhlo.ui32_v1>
  %0 = "stablehlo.partition_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-LABEL: "op_partition_id"

func.func @op_reshape(%arg0: tensor<16xf32>) -> tensor<4x4xf32> {
  // CHECK: "vhlo.reshape_v1"(%arg0) : (!vhlo.tensor_v1<16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<4x4x!vhlo.f32_v1>
  %0 = "stablehlo.reshape"(%arg0) : (tensor<16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}
// CHECK-LABEL: "op_reshape"

func.func @op_return(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
  //      CHECK: "vhlo.case_v1"(%arg0) ({
  // CHECK-NEXT:   "vhlo.return_v1"(%arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  // CHECK-NEXT: }) : (!vhlo.tensor_v1<!vhlo.i32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.case"(%arg0) ({
    "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_return"

func.func @op_reverse(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //      CHECK: "vhlo.reverse_v1"(%arg0) {
  // CHECK-SAME:   dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x!vhlo.f32_v1>
  %0 = "stablehlo.reverse"(%arg0) {
    dimensions = dense<0> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_reverse"

func.func @op_rng_bit_generator(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  //      CHECK: "vhlo.rng_bit_generator_v1"(%arg0) {
  // CHECK-SAME:   rng_algorithm = #vhlo<rng_algorithm_v1 PHILOX>
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>)
  %0:2 = "stablehlo.rng_bit_generator"(%arg0) {
    rng_algorithm = #stablehlo<rng_algorithm PHILOX>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}
// CHECK-LABEL: "op_rng_bit_generator"

func.func @op_rng(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<?xindex>) -> tensor<f32> {
  //      CHECK: "vhlo.rng_v1"(%arg0, %arg1, %arg2) {
  // CHECK-SAME:   rng_distribution = #vhlo<rng_distribution_v1 NORMAL>
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<?x!vhlo.index_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.rng"(%arg0, %arg1, %arg2) {
    rng_distribution = #stablehlo<rng_distribution NORMAL>
  } : (tensor<f32>, tensor<f32>, tensor<?xindex>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_rng"

func.func @op_round_nearest_afz(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.round_nearest_afz_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.round_nearest_afz"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_round_nearest_afz"

func.func @op_round_nearest_even(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.round_nearest_even_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.round_nearest_even"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_round_nearest_even"

func.func @op_rsqrt(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.rsqrt_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.rsqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_rsqrt"

func.func @op_scatter(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>, %arg2: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
  //      CHECK: "vhlo.scatter_v1"(%arg0, %arg1, %arg2) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG3:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG4:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  // CHECK-NEXT:     %[[VAL1:.*]] = "vhlo.add_v1"(%[[ARG3]], %[[ARG4]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  // CHECK-NEXT:     "vhlo.return_v1"(%[[VAL1]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   index_vector_dim = #vhlo.integer_v1<1 : i64>,
  // CHECK-SAME:   indices_are_sorted = #vhlo.bool_v1<true>,
  // CHECK-SAME:   inserted_window_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   scatter_dims_to_operand_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
  // CHECK-SAME:   unique_indices = #vhlo.bool_v1<true>,
  // CHECK-SAME:   update_window_dims = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<200x100x300x!vhlo.f32_v1>, !vhlo.tensor_v1<10x2x!vhlo.i32_v1>, !vhlo.tensor_v1<10x300x!vhlo.f32_v1>) -> !vhlo.tensor_v1<200x100x300x!vhlo.f32_v1>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}
// CHECK-LABEL: "op_scatter"

func.func @op_select_and_scatter(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<12x13x13x66xf32>, %arg2: tensor<f32>) -> tensor<10x24x24x64xf32> {
  //      CHECK: "vhlo.select_and_scatter_v1"(%arg0, %arg1, %arg2) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG31:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG41:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  // CHECK-NEXT:     %[[VAL11:.*]] = "vhlo.compare_v1"(%[[ARG31]], %[[ARG41]]) {compare_type = #vhlo<comparison_type_v1 TOTALORDER>, comparison_direction = #vhlo<comparison_direction_v1 GE>} : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  // CHECK-NEXT:     "vhlo.return_v1"(%[[VAL11]]) : (!vhlo.tensor_v1<!vhlo.bool_v1>) -> ()
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG32:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG42:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  // CHECK-NEXT:     %[[VAL12:.*]] = "vhlo.add_v1"(%[[ARG32]], %[[ARG42]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  // CHECK-NEXT:     "vhlo.return_v1"(%[[VAL12]]) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   padding = #vhlo.tensor_v1<dense<1> : tensor<4x2xi64>>,
  // CHECK-SAME:   window_dimensions = #vhlo.tensor_v1<dense<[1, 2, 2, 1]> : tensor<4xi64>>,
  // CHECK-SAME:   window_strides = #vhlo.tensor_v1<dense<[1, 2, 2, 1]> : tensor<4xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<10x24x24x64x!vhlo.f32_v1>, !vhlo.tensor_v1<12x13x13x66x!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<10x24x24x64x!vhlo.f32_v1>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "stablehlo.compare"(%arg3, %arg4) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    padding = dense<1> : tensor<4x2xi64>
  } : (tensor<10x24x24x64xf32>, tensor<12x13x13x66xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  func.return %0 : tensor<10x24x24x64xf32>
}
// CHECK-LABEL: "op_select_and_scatter"

func.func @op_select(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.select_v1"(%arg0, %arg1, %arg2) : (!vhlo.tensor_v1<!vhlo.bool_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_select"

func.func @op_send(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  //      CHECK: "vhlo.send_v1"(%arg0, %arg1) {
  // CHECK-SAME:   channel_id = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   channel_type = #vhlo.integer_v1<0 : i64>,
  // CHECK-SAME:   is_host_transfer = #vhlo.bool_v1<true>
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.token_v1) -> !vhlo.token_v1
  %0 = "stablehlo.send"(%arg0, %arg1) {
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
    is_host_transfer = true
  } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}
// CHECK-LABEL: "op_send"

func.func @op_set_dimension_size(%arg0: tensor<?xf32>, %arg1: tensor<i32>) -> tensor<16xf32> {
  //      CHECK: "vhlo.set_dimension_size_v1"(%arg0, %arg1) {
  // CHECK-SAME:   dimension = #vhlo.integer_v1<0 : i64>
  // CHECK-SAME: } : (!vhlo.tensor_v1<?x!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.i32_v1>) -> !vhlo.tensor_v1<16x!vhlo.f32_v1>
  %0 = "stablehlo.set_dimension_size"(%arg0, %arg1) {
    dimension = 0 : i64
  } : (tensor<?xf32>, tensor<i32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_set_dimension_size"

func.func @op_shift_left(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "vhlo.shift_left_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.i32_v1>, !vhlo.tensor_v1<!vhlo.i32_v1>) -> !vhlo.tensor_v1<!vhlo.i32_v1>
  %0 = "stablehlo.shift_left"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_shift_left"

func.func @op_shift_right_arithmetic(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "vhlo.shift_right_arithmetic_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.i32_v1>, !vhlo.tensor_v1<!vhlo.i32_v1>) -> !vhlo.tensor_v1<!vhlo.i32_v1>
  %0 = "stablehlo.shift_right_arithmetic"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_shift_right_arithmetic"

func.func @op_shift_right_logical(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "vhlo.shift_right_logical_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.i32_v1>, !vhlo.tensor_v1<!vhlo.i32_v1>) -> !vhlo.tensor_v1<!vhlo.i32_v1>
  %0 = "stablehlo.shift_right_logical"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_shift_right_logical"

func.func @op_sign(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.sign_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.sign"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_sign"

func.func @op_sine(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.sine_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.sine"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_sine"

func.func @op_slice(%arg0: tensor<16xf32>) -> tensor<4xf32> {
  //      CHECK: "vhlo.slice_v1"(%arg0) {
  // CHECK-SAME:   limit_indices = #vhlo.tensor_v1<dense<4> : tensor<1xi64>>,
  // CHECK-SAME:   start_indices = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
  // CHECK-SAME:   strides = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<4x!vhlo.f32_v1>
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = dense<0> : tensor<1xi64>,
    limit_indices = dense<4> : tensor<1xi64>,
    strides = dense<1> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
// CHECK-LABEL: "op_slice"

func.func @op_sort(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //      CHECK: "vhlo.sort_v1"(%arg0) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>, %[[ARG2:arg.*]]: !vhlo.tensor_v1<!vhlo.f32_v1>):
  // CHECK-NEXT:     %[[VAL1:.*]] = "vhlo.compare_v1"(%[[ARG1]], %[[ARG2]]) {compare_type = #vhlo<comparison_type_v1 FLOAT>, comparison_direction = #vhlo<comparison_direction_v1 GT>} : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  // CHECK-NEXT:     "vhlo.return_v1"(%[[VAL1]]) : (!vhlo.tensor_v1<!vhlo.bool_v1>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   dimension = #vhlo.integer_v1<0 : i64>
  // CHECK-SAME:   is_stable = #vhlo.bool_v1<true>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x!vhlo.f32_v1>
  %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.compare"(%arg1, %arg2) {compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }) {
    dimension = 0 : i64,
    is_stable = true
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_sort"

func.func @op_sqrt(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.sqrt_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.sqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_sqrt"

func.func @op_subtract(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.subtract_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.subtract"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_subtract"

func.func @op_tanh(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.tanh_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.tanh"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_tanh"

func.func @op_torch_index_select(%arg0: tensor<5x1x5xf32>, %arg1: tensor<2xi32>) ->  tensor<2x1x5xf32> {
  //      CHECK: "vhlo.torch_index_select_v1"(%arg0, %arg1) {
  // CHECK-SAME:   batch_dims = #vhlo.integer_v1<0 : i64>
  // CHECK-SAME:   dim = #vhlo.integer_v1<0 : i64>
  // CHECK-SAME: } : (!vhlo.tensor_v1<5x1x5x!vhlo.f32_v1>, !vhlo.tensor_v1<2x!vhlo.i32_v1>) -> !vhlo.tensor_v1<2x1x5x!vhlo.f32_v1>
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
    dim = 0 : i64,
    batch_dims = 0 : i64
  } : (tensor<5x1x5xf32>, tensor<2xi32>) -> tensor<2x1x5xf32>
  func.return %0 : tensor<2x1x5xf32>
}
// CHECK-LABEL: "op_torch_index_select"

func.func @op_trace(%arg0: tensor<f32>) {
  //      CHECK: "vhlo.trace_v1"(%arg0) {
  // CHECK-SAME:   tag = #vhlo.string_v1<"foo">
  // CHECK-SAME: } : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
  "stablehlo.trace"(%arg0) {
    tag = "foo"
  } : (tensor<f32>) -> ()
  func.return
}
// CHECK-LABEL: "op_trace"

func.func @op_transpose(%arg0: tensor<16x8xf32>) ->  tensor<8x16xf32> {
  //      CHECK: "vhlo.transpose_v1"(%arg0) {
  // CHECK-SAME:   permutation = #vhlo.tensor_v1<dense<[1, 0]> : tensor<2xi64>>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x8x!vhlo.f32_v1>) -> !vhlo.tensor_v1<8x16x!vhlo.f32_v1>
  %0 = "stablehlo.transpose"(%arg0) {
    permutation = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<16x8xf32>) -> tensor<8x16xf32>
  func.return %0 : tensor<8x16xf32>
}
// CHECK-LABEL: "op_transpose"

func.func @op_triangular_solve(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  //      CHECK: "vhlo.triangular_solve_v1"(%arg0, %arg1) {
  // CHECK-SAME:   left_side = #vhlo.bool_v1<true>,
  // CHECK-SAME:   lower = #vhlo.bool_v1<true>,
  // CHECK-SAME:   transpose_a = #vhlo<transpose_v1 NO_TRANSPOSE>,
  // CHECK-SAME:   unit_diagonal = #vhlo.bool_v1<true>
  // CHECK-SAME: } : (!vhlo.tensor_v1<16x16x!vhlo.f32_v1>, !vhlo.tensor_v1<16x16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<16x16x!vhlo.f32_v1>
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "op_triangular_solve"

func.func @op_tuple(%arg0: tensor<f32>) -> tuple<tensor<f32>> {
  // CHECK: "vhlo.tuple_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tuple_v1<!vhlo.tensor_v1<!vhlo.f32_v1>>
  %0 = "stablehlo.tuple"(%arg0) : (tensor<f32>) -> tuple<tensor<f32>>
  func.return %0 : tuple<tensor<f32>>
}
// CHECK-LABEL: "op_tuple"

func.func @op_unary_einsum(%arg0: tensor<8x16xf32>) -> tensor<8xf32> {
  //      CHECK: "vhlo.unary_einsum_v1"(%arg0) {
  // CHECK-SAME:   einsum_config = #vhlo.string_v1<"ab->a">
  // CHECK-SAME: } : (!vhlo.tensor_v1<8x16x!vhlo.f32_v1>) -> !vhlo.tensor_v1<8x!vhlo.f32_v1>
  %0 = "stablehlo.unary_einsum"(%arg0) {
    einsum_config = "ab->a"
  } : (tensor<8x16xf32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}
// CHECK-LABEL: "op_unary_einsum"

func.func @op_uniform_dequantize(%arg0: tensor<!quant.uniform<i8:f32, 34.0:16>>) -> tensor<f32> {
  // CHECK: "vhlo.uniform_dequantize_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.quant_v1<!vhlo.i8_v1:!vhlo.f32_v1, 3.400000e+01:16, -128:127, 1>>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.uniform_dequantize"(%arg0) : (tensor<!quant.uniform<i8:f32, 34.0:16>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_uniform_dequantize"

func.func @op_uniform_quantize(%arg0: tensor<f32>) -> tensor<!quant.uniform<i8:f32, 34.0:16>> {
  // CHECK: "vhlo.uniform_quantize_v1"(%arg0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.quant_v1<!vhlo.i8_v1:!vhlo.f32_v1, 3.400000e+01:16, -128:127, 1>>
  %0 = "stablehlo.uniform_quantize"(%arg0) : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 34.0:16>>
  func.return %0 : tensor<!quant.uniform<i8:f32, 34.0:16>>
}
// CHECK-LABEL: "op_uniform_quantize"

func.func @op_while(%arg0: tensor<i1>) -> tensor<i1> {
  //      CHECK: "vhlo.while_v1"(%arg0) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: !vhlo.tensor_v1<!vhlo.bool_v1>):
  // CHECK-NEXT:     "vhlo.return_v1"(%[[ARG1]]) : (!vhlo.tensor_v1<!vhlo.bool_v1>) -> ()
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: !vhlo.tensor_v1<!vhlo.bool_v1>)
  // CHECK-NEXT:     "vhlo.return_v1"(%[[ARG1]]) : (!vhlo.tensor_v1<!vhlo.bool_v1>) -> ()
  // CHECK-NEXT: }) : (!vhlo.tensor_v1<!vhlo.bool_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  %0 = "stablehlo.while"(%arg0) ({
    ^bb0(%arg1: tensor<i1>):
      "stablehlo.return"(%arg1) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<i1>):
      "stablehlo.return"(%arg1) : (tensor<i1>) -> ()
  }) : (tensor<i1>) -> tensor<i1>
  func.return %0: tensor<i1>
}
// CHECK-LABEL: "op_while"

func.func @op_xor(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "vhlo.xor_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.bool_v1>, !vhlo.tensor_v1<!vhlo.bool_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  %0 = "stablehlo.xor"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_xor"

// ============ TYPES ============

func.func @type_i1(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "vhlo.and_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.bool_v1>, !vhlo.tensor_v1<!vhlo.bool_v1>) -> !vhlo.tensor_v1<!vhlo.bool_v1>
  %0 = "stablehlo.and"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "type_i1"

func.func @type_i4(%arg0: tensor<i4>, %arg1: tensor<i4>) -> tensor<i4> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.i4_v1>, !vhlo.tensor_v1<!vhlo.i4_v1>) -> !vhlo.tensor_v1<!vhlo.i4_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i4>, tensor<i4>) -> tensor<i4>
  func.return %0 : tensor<i4>
}
// CHECK-LABEL: "type_i4"

func.func @type_i8(%arg0: tensor<i8>, %arg1: tensor<i8>) -> tensor<i8> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.i8_v1>, !vhlo.tensor_v1<!vhlo.i8_v1>) -> !vhlo.tensor_v1<!vhlo.i8_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
  func.return %0 : tensor<i8>
}
// CHECK-LABEL: "type_i8"

func.func @type_i16(%arg0: tensor<i16>, %arg1: tensor<i16>) -> tensor<i16> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.i16_v1>, !vhlo.tensor_v1<!vhlo.i16_v1>) -> !vhlo.tensor_v1<!vhlo.i16_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
  func.return %0 : tensor<i16>
}
// CHECK-LABEL: "type_i16"

func.func @type_i32(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.i32_v1>, !vhlo.tensor_v1<!vhlo.i32_v1>) -> !vhlo.tensor_v1<!vhlo.i32_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "type_i32"

func.func @type_i64(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.i64_v1>, !vhlo.tensor_v1<!vhlo.i64_v1>) -> !vhlo.tensor_v1<!vhlo.i64_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  func.return %0 : tensor<i64>
}
// CHECK-LABEL: "type_i64"

func.func @type_ui4(%arg0: tensor<ui4>, %arg1: tensor<ui4>) -> tensor<ui4> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.ui4_v1>, !vhlo.tensor_v1<!vhlo.ui4_v1>) -> !vhlo.tensor_v1<!vhlo.ui4_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui4>, tensor<ui4>) -> tensor<ui4>
  func.return %0 : tensor<ui4>
}
// CHECK-LABEL: "type_ui4"

func.func @type_ui8(%arg0: tensor<ui8>, %arg1: tensor<ui8>) -> tensor<ui8> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.ui8_v1>, !vhlo.tensor_v1<!vhlo.ui8_v1>) -> !vhlo.tensor_v1<!vhlo.ui8_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
  func.return %0 : tensor<ui8>
}
// CHECK-LABEL: "type_ui8"

func.func @type_ui16(%arg0: tensor<ui16>, %arg1: tensor<ui16>) -> tensor<ui16> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.ui16_v1>, !vhlo.tensor_v1<!vhlo.ui16_v1>) -> !vhlo.tensor_v1<!vhlo.ui16_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
  func.return %0 : tensor<ui16>
}
// CHECK-LABEL: "type_ui16"

func.func @type_ui32(%arg0: tensor<ui32>, %arg1: tensor<ui32>) -> tensor<ui32> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.ui32_v1>, !vhlo.tensor_v1<!vhlo.ui32_v1>) -> !vhlo.tensor_v1<!vhlo.ui32_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-LABEL: "type_ui32"

func.func @type_ui64(%arg0: tensor<ui64>, %arg1: tensor<ui64>) -> tensor<ui64> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.ui64_v1>, !vhlo.tensor_v1<!vhlo.ui64_v1>) -> !vhlo.tensor_v1<!vhlo.ui64_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
  func.return %0 : tensor<ui64>
}
// CHECK-LABEL: "type_ui64"

func.func @type_f8E4M3FN(%arg0: tensor<f8E4M3FN>, %arg1: tensor<f8E4M3FN>) -> tensor<f8E4M3FN> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f8E4M3FN_v1>, !vhlo.tensor_v1<!vhlo.f8E4M3FN_v1>) -> !vhlo.tensor_v1<!vhlo.f8E4M3FN_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<f8E4M3FN>
  func.return %0 : tensor<f8E4M3FN>
}
// CHECK-LABEL: "type_f8E4M3FN"

func.func @type_f8E5M2(%arg0: tensor<f8E5M2>, %arg1: tensor<f8E5M2>) -> tensor<f8E5M2> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f8E5M2_v1>, !vhlo.tensor_v1<!vhlo.f8E5M2_v1>) -> !vhlo.tensor_v1<!vhlo.f8E5M2_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E5M2>, tensor<f8E5M2>) -> tensor<f8E5M2>
  func.return %0 : tensor<f8E5M2>
}
// CHECK-LABEL: "type_f8E5M2"

func.func @type_bf16(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.bf16_v1>, !vhlo.tensor_v1<!vhlo.bf16_v1>) -> !vhlo.tensor_v1<!vhlo.bf16_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
  func.return %0 : tensor<bf16>
}
// CHECK-LABEL: "type_bf16"

func.func @type_f16(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f16_v1>, !vhlo.tensor_v1<!vhlo.f16_v1>) -> !vhlo.tensor_v1<!vhlo.f16_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
  func.return %0 : tensor<f16>
}
// CHECK-LABEL: "type_f16"

func.func @type_f32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f32_v1>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "type_f32"

func.func @type_f64(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.f64_v1>, !vhlo.tensor_v1<!vhlo.f64_v1>) -> !vhlo.tensor_v1<!vhlo.f64_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  func.return %0 : tensor<f64>
}
// CHECK-LABEL: "type_f64"

func.func @type_complex_f32(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>) -> tensor<complex<f32>> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.complex_v1<!vhlo.f32_v1>>, !vhlo.tensor_v1<!vhlo.complex_v1<!vhlo.f32_v1>>) -> !vhlo.tensor_v1<!vhlo.complex_v1<!vhlo.f32_v1>>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %0 : tensor<complex<f32>>
}
// CHECK-LABEL: "type_complex_f32"

func.func @type_complex_f64(%arg0: tensor<complex<f64>>, %arg1: tensor<complex<f64>>) -> tensor<complex<f64>> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.complex_v1<!vhlo.f64_v1>>, !vhlo.tensor_v1<!vhlo.complex_v1<!vhlo.f64_v1>>) -> !vhlo.tensor_v1<!vhlo.complex_v1<!vhlo.f64_v1>>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
  func.return %0 : tensor<complex<f64>>
}
// CHECK-LABEL: "type_complex_f64"

func.func @type_dynamism_ranked(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: "vhlo.abs_v1"(%arg0) : (!vhlo.tensor_v1<?x!vhlo.f32_v1>) -> !vhlo.tensor_v1<?x!vhlo.f32_v1>
  %0 = "stablehlo.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: "type_dynamism_ranked"

func.func @type_dynamism_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "vhlo.abs_v1"(%arg0) : (!vhlo.unranked_tensor_v1<!vhlo.f32_v1>) -> !vhlo.unranked_tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.abs"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
// CHECK-LABEL: "type_dynamism_unranked"

func.func @type_quantization(%arg0: tensor<!quant.uniform<i8:f32, 34.0:16>>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "vhlo.add_v1"(%arg0, %arg1) : (!vhlo.tensor_v1<!vhlo.quant_v1<!vhlo.i8_v1:!vhlo.f32_v1, 3.400000e+01:16, -128:127, 1>>, !vhlo.tensor_v1<!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<!quant.uniform<i8:f32, 34.0:16>>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "type_quantization"

func.func @type_token_callee(%arg0: !stablehlo.token) -> !stablehlo.token {
  // CHECK: "vhlo.return_v1"(%arg0) : (!vhlo.token_v1) -> ()
  return %arg0 : !stablehlo.token
}
//       CHECK: function_type = #vhlo.type_v1<!vhlo.func_v1<(!vhlo.token_v1) -> !vhlo.token_v1>>
// CHECK-LABEL: "type_token_callee"

func.func @type_token_caller(%arg0: !stablehlo.token) -> !stablehlo.token {
  // CHECK:      "vhlo.call_v1"(%arg0) {callee = #vhlo.string_v1<"type_token_callee">}
  // CHECK-SAME: (!vhlo.token_v1) -> !vhlo.token_v1
  %0 = func.call @type_token_callee(%arg0) : (!stablehlo.token) -> !stablehlo.token
  return %0 : !stablehlo.token
}
//       CHECK: function_type = #vhlo.type_v1<!vhlo.func_v1<(!vhlo.token_v1) -> !vhlo.token_v1>>
// CHECK-LABEL: "type_token_caller"

func.func @type_tuple(%arg0: tuple<tensor<f32>>) -> tuple<!stablehlo.token> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo"
  // CHECK: (!vhlo.tuple_v1<!vhlo.tensor_v1<!vhlo.f32_v1>>) -> !vhlo.tuple_v1<!vhlo.token_v1>
  } : (tuple<tensor<f32>>) -> tuple<!stablehlo.token>
  return %0 : tuple<!stablehlo.token>
}
// CHECK-LABEL: "type_tuple"
