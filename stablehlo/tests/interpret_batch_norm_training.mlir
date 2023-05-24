// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @batch_norm_training() {
  %operand = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]],
                                       [[3.0, 4.0], [1.0, 2.0]]]> : tensor<2x2x2xf64>
  %scale = stablehlo.constant dense<[1.0, 1.0]> : tensor<2xf64>
  %offset = stablehlo.constant dense<[1.0, 1.0]> : tensor<2xf64>
  %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%operand, %scale, %offset) {
    epsilon = 0.0 : f32,
    feature_index = 2 : i64
  } : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>) ->
      (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>)
  check.expect_almost_eq_const %output, dense<[[[0.0, 0.0], [2.0, 2.0]],
                                               [[2.0, 2.0], [0.0, 0.0]]]> : tensor<2x2x2xf64>
  check.expect_almost_eq_const %batch_mean, dense<[2.0, 3.0]> : tensor<2xf64>
  check.expect_almost_eq_const %batch_var, dense<[1.0, 1.0]> : tensor<2xf64>
  func.return
}
