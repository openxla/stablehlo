// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @reshape_op_test_si32() {
  %0 = stablehlo.constant dense<[[1,2,3,4,5,6]]> : tensor<1x6xi32>
  %1 = stablehlo.reshape %0 : (tensor<1x6xi32>) -> tensor<6xi32>
  check.expect_eq_const %1, dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  func.return
}

// -----

func.func @reshape_op_test_si32() {
  %0 = stablehlo.constant dense<[1,2,3,4,5,6]> : tensor<6xi32>
  %1 = stablehlo.reshape %0 : (tensor<6xi32>) -> tensor<2x3xi32>
  check.expect_eq_const %1, dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  func.return
}

// -----

func.func @reshape_op_test_si32() {
  %0 = stablehlo.constant dense<[[1,2,3],[4,5,6]]> : tensor<2x3xi32>
  %1 = stablehlo.reshape %0 : (tensor<2x3xi32>) -> tensor<3x2xi32>
  check.expect_eq_const %1, dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  func.return
}

// -----

func.func @reshape_op_test_si32() {
  %0 = stablehlo.constant dense<[[1,2],[3,4],[5,6]]> : tensor<3x2xi32>
  %1 = stablehlo.reshape %0 : (tensor<3x2xi32>) -> tensor<6xi32>
  check.expect_eq_const %1, dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  func.return
}
