// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @transpose_op_test_si32() {
  %0 = stablehlo.constant dense<[[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]> : tensor<2x3x2xi32>
  %1 = "stablehlo.transpose"(%0) {permutation = dense<[1,0,2]> : tensor<3xi64>} : (tensor<2x3x2xi32>) -> tensor<3x2x2xi32>
  check.expect_eq_const %1, dense<[[[1, 2], [7, 8]], [[3, 4], [9, 10]], [[5, 6], [11, 12]]]> : tensor<3x2x2xi32>
  func.return
}

// -----

func.func @transpose_op_test_si32() {
  %0 = stablehlo.constant dense<[[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]> : tensor<2x3x2xi32>
  %1 = "stablehlo.transpose"(%0) {permutation = dense<[2,1,0]> : tensor<3xi64>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  check.expect_eq_const %1, dense<[[[1, 7], [3, 9], [5, 11]], [[2, 8], [4, 10], [6, 12]]]> : tensor<2x3x2xi32>
  func.return
}

// -----

func.func @transpose_op_test_si32() {
  %0 = stablehlo.constant dense<[[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]> : tensor<2x3x2xi32>
  %1 = "stablehlo.transpose"(%0) {permutation = dense<[2,1,0]> : tensor<3xi64>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  %2 = "stablehlo.transpose"(%1) {permutation = dense<[2,1,0]> : tensor<3xi64>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  check.expect_eq_const %2, dense<[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]> : tensor<2x3x2xi32>
  func.return
}
