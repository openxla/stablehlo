// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si4
func.func @dot_general_op_test_si4() -> tensor<2x2x2xi4> {
  %0 = stablehlo.constant dense<[[[-3, -2], [-1, 0]], [[1, 2], [3, 4]]]> : tensor<2x2x2xi4>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi4>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi4>, tensor<2x2x2xi4>) -> tensor<2x2x2xi4>
  func.return %2 : tensor<2x2x2xi4>
  // CHECK-NEXT: tensor<2x2x2xi4>
  // CHECK-NEXT: -3 : i4
  // CHECK-NEXT: -2 : i4
  // CHECK-NEXT: -1 : i4
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 2 : i4
  // CHECK-NEXT: 3 : i4
  // CHECK-NEXT: 4 : i4
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_ui4
func.func @dot_general_op_test_ui4() -> tensor<2x2x2xui4> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xui4>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xui4>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xui4>, tensor<2x2x2xui4>) -> tensor<2x2x2xui4>
  func.return %2 : tensor<2x2x2xui4>
  // CHECK-NEXT: tensor<2x2x2xui4>
  // CHECK-NEXT: 1 : ui4
  // CHECK-NEXT: 2 : ui4
  // CHECK-NEXT: 3 : ui4
  // CHECK-NEXT: 4 : ui4
  // CHECK-NEXT: 5 : ui4
  // CHECK-NEXT: 6 : ui4
  // CHECK-NEXT: 7 : ui4
  // CHECK-NEXT: 8 : ui4
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si8
func.func @dot_general_op_test_si8() -> tensor<2x2x2xi8> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi8>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi8>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi8>, tensor<2x2x2xi8>) -> tensor<2x2x2xi8>
  func.return %2 : tensor<2x2x2xi8>
  // CHECK-NEXT: tensor<2x2x2xi8>
  // CHECK-NEXT: 1 : i8
  // CHECK-NEXT: 2 : i8
  // CHECK-NEXT: 3 : i8
  // CHECK-NEXT: 4 : i8
  // CHECK-NEXT: 5 : i8
  // CHECK-NEXT: 6 : i8
  // CHECK-NEXT: 7 : i8
  // CHECK-NEXT: 8 : i8
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_ui8
func.func @dot_general_op_test_ui8() -> tensor<2x2x2xui8> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xui8>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xui8>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xui8>, tensor<2x2x2xui8>) -> tensor<2x2x2xui8>
  func.return %2 : tensor<2x2x2xui8>
  // CHECK-NEXT: tensor<2x2x2xui8>
  // CHECK-NEXT: 1 : ui8
  // CHECK-NEXT: 2 : ui8
  // CHECK-NEXT: 3 : ui8
  // CHECK-NEXT: 4 : ui8
  // CHECK-NEXT: 5 : ui8
  // CHECK-NEXT: 6 : ui8
  // CHECK-NEXT: 7 : ui8
  // CHECK-NEXT: 8 : ui8
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si16
func.func @dot_general_op_test_si16() -> tensor<2x2x2xi16> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi16>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi16>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi16>, tensor<2x2x2xi16>) -> tensor<2x2x2xi16>
  func.return %2 : tensor<2x2x2xi16>
  // CHECK-NEXT: tensor<2x2x2xi16>
  // CHECK-NEXT: 1 : i16
  // CHECK-NEXT: 2 : i16
  // CHECK-NEXT: 3 : i16
  // CHECK-NEXT: 4 : i16
  // CHECK-NEXT: 5 : i16
  // CHECK-NEXT: 6 : i16
  // CHECK-NEXT: 7 : i16
  // CHECK-NEXT: 8 : i16
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_ui16
func.func @dot_general_op_test_ui16() -> tensor<2x2x2xui16> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xui16>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xui16>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xui16>, tensor<2x2x2xui16>) -> tensor<2x2x2xui16>
  func.return %2 : tensor<2x2x2xui16>
  // CHECK-NEXT: tensor<2x2x2xui16>
  // CHECK-NEXT: 1 : ui16
  // CHECK-NEXT: 2 : ui16
  // CHECK-NEXT: 3 : ui16
  // CHECK-NEXT: 4 : ui16
  // CHECK-NEXT: 5 : ui16
  // CHECK-NEXT: 6 : ui16
  // CHECK-NEXT: 7 : ui16
  // CHECK-NEXT: 8 : ui16
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si32
func.func @dot_general_op_test_si32() -> tensor<2x2x2xi32> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi32>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi32>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi32>, tensor<2x2x2xi32>) -> tensor<2x2x2xi32>
  func.return %2 : tensor<2x2x2xi32>
  // CHECK-NEXT: tensor<2x2x2xi32>
  // CHECK-NEXT: 1 : i32
  // CHECK-NEXT: 2 : i32
  // CHECK-NEXT: 3 : i32
  // CHECK-NEXT: 4 : i32
  // CHECK-NEXT: 5 : i32
  // CHECK-NEXT: 6 : i32
  // CHECK-NEXT: 7 : i32
  // CHECK-NEXT: 8 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_ui32
func.func @dot_general_op_test_ui32() -> tensor<2x2x2xui32> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xui32>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xui32>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xui32>, tensor<2x2x2xui32>) -> tensor<2x2x2xui32>
  func.return %2 : tensor<2x2x2xui32>
  // CHECK-NEXT: tensor<2x2x2xui32>
  // CHECK-NEXT: 1 : ui32
  // CHECK-NEXT: 2 : ui32
  // CHECK-NEXT: 3 : ui32
  // CHECK-NEXT: 4 : ui32
  // CHECK-NEXT: 5 : ui32
  // CHECK-NEXT: 6 : ui32
  // CHECK-NEXT: 7 : ui32
  // CHECK-NEXT: 8 : ui32
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si64
func.func @dot_general_op_test_si64() -> tensor<2x2x2xi64> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
  func.return %2 : tensor<2x2x2xi64>
  // CHECK-NEXT: tensor<2x2x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 4 : i64
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: 6 : i64
  // CHECK-NEXT: 7 : i64
  // CHECK-NEXT: 8 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si64_empty_dims
func.func @dot_general_op_test_si64_empty_dims() -> tensor<2x2x2x2xi64> {
  %0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
  %1 = stablehlo.constant dense<[[1, 0], [0, 1]]> : tensor<2x2xi64>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [],
      rhs_contracting_dimensions = []
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2x2x2xi64>
  func.return %2 : tensor<2x2x2x2xi64>
  // CHECK-NEXT: tensor<2x2x2x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 4 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 4 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si64_empty_batching_dims_1
func.func @dot_general_op_test_si64_empty_batching_dims_1() -> tensor<2x2xi64> {
  %0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
  %1 = stablehlo.constant dense<[[1, 0], [0, 1]]> : tensor<2x2xi64>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
  func.return %2 : tensor<2x2xi64>
  // CHECK-NEXT: tensor<2x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 4 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si64_empty_batching_dims_2
func.func @dot_general_op_test_si64_empty_batching_dims_2() -> tensor<i64> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [0, 1, 2],
      rhs_contracting_dimensions = [2, 0, 1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<i64>
  func.return %2 : tensor<i64>
  // CHECK-NEXT: tensor<i64>
  // CHECK-NEXT: 18 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si64_empty_contracting_dims_1
func.func @dot_general_op_test_si64_empty_contracting_dims_1() -> tensor<2x2x2xi64> {
  %0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
  %1 = stablehlo.constant dense<[[1, 0], [0, 1]]> : tensor<2x2xi64>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [],
      rhs_contracting_dimensions = []
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2x2xi64>
  func.return %2 : tensor<2x2x2xi64>
  // CHECK-NEXT: tensor<2x2x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 4 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si64_empty_contracting_dims_2
func.func @dot_general_op_test_si64_empty_contracting_dims_2() -> tensor<2x2x2xi64> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1, 2],
      rhs_batching_dimensions = [2, 0, 1],
      lhs_contracting_dimensions = [],
      rhs_contracting_dimensions = []
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
  func.return %2 : tensor<2x2x2xi64>
  // CHECK-NEXT: tensor<2x2x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 6 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 8 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si64_mix_1
func.func @dot_general_op_test_si64_mix_1() -> tensor<2x2xi64> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 2],
      rhs_batching_dimensions = [2, 1],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2xi64>
  func.return %2 : tensor<2x2xi64>
  // CHECK-NEXT: tensor<2x2xi64>
  // CHECK-NEXT: 4 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 14 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_si64_mix_2
func.func @dot_general_op_test_si64_mix_2() -> tensor<2xi64> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1, 2],
      rhs_contracting_dimensions = [2, 1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2xi64>
  func.return %2 : tensor<2xi64>
  // CHECK-NEXT: tensor<2xi64>
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: 13 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_ui64
func.func @dot_general_op_test_ui64() -> tensor<2x2x2xui64> {
  %0 = stablehlo.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xui64>
  %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xui64>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xui64>, tensor<2x2x2xui64>) -> tensor<2x2x2xui64>
  func.return %2 : tensor<2x2x2xui64>
  // CHECK-NEXT: tensor<2x2x2xui64>
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 5 : ui64
  // CHECK-NEXT: 6 : ui64
  // CHECK-NEXT: 7 : ui64
  // CHECK-NEXT: 8 : ui64
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_i1
func.func @dot_general_op_test_i1() -> tensor<2x2x2xi1> {
  %0 = stablehlo.constant dense<[[[true, true], [true, true]], [[false, false], [false, false]]]> : tensor<2x2x2xi1>
  %1 = stablehlo.constant dense<[[[true, false], [false, true]], [[true, false], [false, true]]]> : tensor<2x2x2xi1>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi1>, tensor<2x2x2xi1>) -> tensor<2x2x2xi1>
  func.return %2 : tensor<2x2x2xi1>
  // CHECK-NEXT: tensor<2x2x2xi1>
  // CHECK-NEXT: true
  // CHECK-NEXT: true
  // CHECK-NEXT: true
  // CHECK-NEXT: true
  // CHECK-NEXT: false
  // CHECK-NEXT: false
  // CHECK-NEXT: false
  // CHECK-NEXT: false
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_bf16
func.func @dot_general_op_test_bf16() -> tensor<2x2x2xbf16> {
  %0 = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xbf16>
  %1 = stablehlo.constant dense<[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]> : tensor<2x2x2xbf16>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>
  func.return %2 : tensor<2x2x2xbf16>
  // CHECK-NEXT: tensor<2x2x2xbf16>
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 2.000000e+00 : bf16
  // CHECK-NEXT: 3.000000e+00 : bf16
  // CHECK-NEXT: 4.000000e+00 : bf16
  // CHECK-NEXT: 5.000000e+00 : bf16
  // CHECK-NEXT: 6.000000e+00 : bf16
  // CHECK-NEXT: 7.000000e+00 : bf16
  // CHECK-NEXT: 8.000000e+00 : bf16
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_f16
func.func @dot_general_op_test_f16() -> tensor<2x2x2xf16> {
  %0 = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf16>
  %1 = stablehlo.constant dense<[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]> : tensor<2x2x2xf16>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xf16>, tensor<2x2x2xf16>) -> tensor<2x2x2xf16>
  func.return %2 : tensor<2x2x2xf16>
  // CHECK-NEXT: tensor<2x2x2xf16>
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 2.000000e+00 : f16
  // CHECK-NEXT: 3.000000e+00 : f16
  // CHECK-NEXT: 4.000000e+00 : f16
  // CHECK-NEXT: 5.000000e+00 : f16
  // CHECK-NEXT: 6.000000e+00 : f16
  // CHECK-NEXT: 7.000000e+00 : f16
  // CHECK-NEXT: 8.000000e+00 : f16
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_f32
func.func @dot_general_op_test_f32() -> tensor<2x2x2xf32> {
  %0 = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf32>
  %1 = stablehlo.constant dense<[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]> : tensor<2x2x2xf32>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  func.return %2 : tensor<2x2x2xf32>
  // CHECK-NEXT: tensor<2x2x2xf32>
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 2.000000e+00 : f32
  // CHECK-NEXT: 3.000000e+00 : f32
  // CHECK-NEXT: 4.000000e+00 : f32
  // CHECK-NEXT: 5.000000e+00 : f32
  // CHECK-NEXT: 6.000000e+00 : f32
  // CHECK-NEXT: 7.000000e+00 : f32
  // CHECK-NEXT: 8.000000e+00 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_f64
func.func @dot_general_op_test_f64() -> tensor<2x2x2xf64> {
  %0 = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf64>
  %1 = stablehlo.constant dense<[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]> : tensor<2x2x2xf64>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xf64>, tensor<2x2x2xf64>) -> tensor<2x2x2xf64>
  func.return %2 : tensor<2x2x2xf64>
  // CHECK-NEXT: tensor<2x2x2xf64>
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 2.000000e+00 : f64
  // CHECK-NEXT: 3.000000e+00 : f64
  // CHECK-NEXT: 4.000000e+00 : f64
  // CHECK-NEXT: 5.000000e+00 : f64
  // CHECK-NEXT: 6.000000e+00 : f64
  // CHECK-NEXT: 7.000000e+00 : f64
  // CHECK-NEXT: 8.000000e+00 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_c64
func.func @dot_general_op_test_c64() -> tensor<2x2x2xcomplex<f32>> {
  %0 = stablehlo.constant dense<[[[(1.0, 0.0), (2.0, 0.0)], [(3.0, 0.0), (4.0, 0.0)]], [[(5.0, 0.0), (6.0, 0.0)], [(7.0, 0.0), (8.0, 0.0)]]]> : tensor<2x2x2xcomplex<f32>>
  %1 = stablehlo.constant dense<[[[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]], [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]]]> : tensor<2x2x2xcomplex<f32>>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xcomplex<f32>>, tensor<2x2x2xcomplex<f32>>) -> tensor<2x2x2xcomplex<f32>>
  func.return %2 : tensor<2x2x2xcomplex<f32>>
  // CHECK-NEXT: tensor<2x2x2xcomplex<f32>>
  // CHECK-NEXT: [1.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [2.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [3.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [4.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [5.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [6.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [7.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [8.000000e+00 : f32, 0.000000e+00 : f32]
}

// -----

// CHECK-LABEL: Evaluated results of function: dot_general_op_test_c128
func.func @dot_general_op_test_c128() -> tensor<2x2x2xcomplex<f64>> {
  %0 = stablehlo.constant dense<[[[(1.0, 0.0), (2.0, 0.0)], [(3.0, 0.0), (4.0, 0.0)]], [[(5.0, 0.0), (6.0, 0.0)], [(7.0, 0.0), (8.0, 0.0)]]]> : tensor<2x2x2xcomplex<f64>>
  %1 = stablehlo.constant dense<[[[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]], [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]]]> : tensor<2x2x2xcomplex<f64>>
  %2 = "stablehlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xcomplex<f64>>, tensor<2x2x2xcomplex<f64>>) -> tensor<2x2x2xcomplex<f64>>
  func.return %2 : tensor<2x2x2xcomplex<f64>>
  // CHECK-NEXT: tensor<2x2x2xcomplex<f64>>
  // CHECK-NEXT: [1.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [2.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [3.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [4.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [5.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [6.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [7.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [8.000000e+00 : f64, 0.000000e+00 : f64]
}
