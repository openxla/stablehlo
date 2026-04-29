// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @triangular_solve_op_test_real_3x3_diagonal_left_lower() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[3.5], [2.0], [1.8]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_left_upper() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[3.5], [2.0], [1.8]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_right_lower() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[3.5, 2.0, 1.8]]> : tensor<1x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_right_upper() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[3.5, 2.0, 1.8]]> : tensor<1x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_left_lower_unit_diagonal() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_left_upper_unit_diagonal() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = false, unit_diagonal = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_right_lower_unit_diagonal() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = true, unit_diagonal = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_right_upper_unit_diagonal() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = false, unit_diagonal = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_left_lower_transpose() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[3.5], [2.0], [1.8]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_left_upper_transpose() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[3.5], [2.0], [1.8]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_right_lower_transpose() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[3.5, 2.0, 1.8]]> : tensor<1x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_right_upper_transpose() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[3.5, 2.0, 1.8]]> : tensor<1x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_left_lower_unit_diagonal_transpose() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = true, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_left_upper_unit_diagonal_transpose() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = false, unit_diagonal = true, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_right_lower_unit_diagonal_transpose() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = true, unit_diagonal = true, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_diagonal_right_upper_unit_diagonal_transpose() {
  %a = stablehlo.constant dense<[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = false, unit_diagonal = true, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_triangular_left_lower() {
  %a = stablehlo.constant dense<[[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[7.0], [-2.0], [-1.5]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_triangular_left_upper() {
  %a = stablehlo.constant dense<[[1.0, 2.0, 4.0], [0.0, 3.0, 5.0], [0.0, 0.0, 6.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[0.666667], [0.166667], [1.5]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_triangular_right_lower() {
  %a = stablehlo.constant dense<[[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[0.666667, 0.166667, 1.5]]> : tensor<1x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_triangular_right_upper() {
  %a = stablehlo.constant dense<[[1.0, 2.0, 4.0], [0.0, 3.0, 5.0], [0.0, 0.0, 6.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[7.0, -2.0, -1.5]]> : tensor<1x3xf32>
  func.return
}

// // -----

func.func @triangular_solve_op_test_real_3x3_triangular_left_lower_transpose() {
  %a = stablehlo.constant dense<[[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[0.666667], [0.166667], [1.5]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_triangular_left_upper_transpose() {
  %a = stablehlo.constant dense<[[1.0, 2.0, 4.0], [0.0, 3.0, 5.0], [0.0, 0.0, 6.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0], [8.0], [9.0]]> : tensor<3x1xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const %result, dense<[[7.0], [-2.0], [-1.5]]> : tensor<3x1xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_triangular_right_lower_transpose() {
  %a = stablehlo.constant dense<[[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[7.0, -2.0, -1.5]]> : tensor<1x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_real_3x3_triangular_right_upper_transpose() {
  %a = stablehlo.constant dense<[[1.0, 2.0, 4.0], [0.0, 3.0, 5.0], [0.0, 0.0, 6.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = false, lower = false, unit_diagonal = false, transpose_a = #stablehlo<transpose TRANSPOSE>} : (tensor<3x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const %result, dense<[[0.666667, 0.166667, 1.5]]> : tensor<1x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_spec_example() {
  %a = stablehlo.constant dense<[[1.0, 0.0, 0.0], [2.0, 4.0, 0.0], [3.0, 5.0, 6.0]]> : tensor<3x3xf32>
  %b = stablehlo.constant dense<[[2.0, 0.0, 0.0], [4.0, 8.0, 0.0], [6.0, 10.0, 12.0]]> : tensor<3x3xf32>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
  check.expect_almost_eq_const %result, dense<[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]> : tensor<3x3xf32>
  func.return
}

// -----

func.func @triangular_solve_op_test_complex_example() {
  %a = stablehlo.constant dense<[[(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)], [(2.0, -2.0), (3.0, 4.0), (0.0, 0.0)], [(4.0, -3.0), (5.0, 5.0), (6.0, -8.0)]]> : tensor<3x3xcomplex<f32>>
  %b = stablehlo.constant dense<[[(7.0, -2.0)], [(8.0, 4.0)], [(9.0, -5.0)]]> : tensor<3x1xcomplex<f32>>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<3x3xcomplex<f32>>, tensor<3x1xcomplex<f32>>) -> tensor<3x1xcomplex<f32>>
  check.expect_almost_eq_const %result, dense<[[(2.5, -4.5)], [(4.32, 0.24)], [(-0.29, -0.77)]]> : tensor<3x1xcomplex<f32>>
  func.return
}

// -----

func.func @triangular_solve_op_test_complex_example_adjoint() {
  %a = stablehlo.constant dense<[[(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)], [(2.0, -2.0), (3.0, 4.0), (0.0, 0.0)], [(4.0, -3.0), (5.0, 5.0), (6.0, -8.0)]]> : tensor<3x3xcomplex<f32>>
  %b = stablehlo.constant dense<[[(7.0, -2.0)], [(8.0, 4.0)], [(9.0, -5.0)]]> : tensor<3x1xcomplex<f32>>
  %result = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose ADJOINT>} : (tensor<3x3xcomplex<f32>>, tensor<3x1xcomplex<f32>>) -> tensor<3x1xcomplex<f32>>

  interpreter.print %result : tensor<3x1xcomplex<f32>>

  check.expect_almost_eq_const %result, dense<[[(7.18, 2.679999)], [(-0.08, 3.16)], [(0.14, -1.02)]]> : tensor<3x1xcomplex<f32>>
  func.return
}
