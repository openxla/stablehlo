// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @dot_general_op_test_si64() {
  %lhs = stablehlo.constant dense<[[[1, 2], [3, 4]],
                                   [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  %rhs = stablehlo.constant dense<[[[1, 0], [0, 1]],
                                   [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>
  %result = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
  check.expect_eq_const %result, dense<[[[1, 2], [3, 4]],
                                        [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  func.return
}

// -----

func.func @dot_general_op_test_ui64() {
  %lhs = stablehlo.constant dense<[[[1, 2], [3, 4]],
                                   [[5, 6], [7, 8]]]> : tensor<2x2x2xui64>
  %rhs = stablehlo.constant dense<[[[1, 0], [0, 1]],
                                   [[1, 0], [0, 1]]]> : tensor<2x2x2xui64>
  %result = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 2],
      rhs_batching_dimensions = [2, 1],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xui64>, tensor<2x2x2xui64>) -> tensor<2x2xui64>
  check.expect_eq_const %result, dense<[[4, 0],
                                        [0, 14]]> : tensor<2x2xui64>
  func.return
}

// -----

func.func @dot_general_op_test_i1() {
  %lhs = stablehlo.constant dense<[[[true, true], [true, true]],
                                   [[false, false], [false, false]]]> : tensor<2x2x2xi1>
  %rhs = stablehlo.constant dense<[[[true, false], [false, true]],
                                   [[true, false], [false, true]]]> : tensor<2x2x2xi1>
  %result = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi1>, tensor<2x2x2xi1>) -> tensor<2x2x2xi1>
  check.expect_eq_const %result, dense<[[[true, true], [true, true]],
                                        [[false, false], [false, false]]]> : tensor<2x2x2xi1>
  func.return
}

// -----

func.func @dot_general_op_test_f64() {
  %lhs = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]],
                                   [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf64>
  %rhs = stablehlo.constant dense<[[[1.0, 0.0], [0.0, 1.0]],
                                   [[1.0, 0.0], [0.0, 1.0]]]> : tensor<2x2x2xf64>
  %result = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xf64>, tensor<2x2x2xf64>) -> tensor<2x2x2xf64>
  check.expect_eq_const %result, dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf64>
  func.return
}

// -----

func.func @dot_general_op_test_c128() {
  %lhs = stablehlo.constant dense<[[[(1.0, 0.0), (2.0, 0.0)], [(3.0, 0.0), (4.0, 0.0)]],
                                   [[(5.0, 0.0), (6.0, 0.0)], [(7.0, 0.0), (8.0, 0.0)]]]> : tensor<2x2x2xcomplex<f64>>
  %rhs = stablehlo.constant dense<[[[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]],
                                   [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]]]> : tensor<2x2x2xcomplex<f64>>
  %result = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xcomplex<f64>>, tensor<2x2x2xcomplex<f64>>) -> tensor<2x2x2xcomplex<f64>>
  check.expect_eq_const %result, dense<[[[(1.0, 0.0), (2.0, 0.0)], [(3.0, 0.0), (4.0, 0.0)]],
                                        [[(5.0, 0.0), (6.0, 0.0)], [(7.0, 0.0), (8.0, 0.0)]]]> : tensor<2x2x2xcomplex<f64>>
  func.return
}
