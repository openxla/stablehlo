// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @batch_norm_inference() {
  %operand = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]],
                                       [[3.0, 4.0], [1.0, 2.0]]]> : tensor<2x2x2xf64>
  %scale = stablehlo.constant dense<[1.0, 1.0]> : tensor<2xf64>
  %offset = stablehlo.constant dense<[1.0, 1.0]> : tensor<2xf64>
  %mean = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf64>
  %variance = stablehlo.constant dense<[1.0, 1.0]> : tensor<2xf64>
  %result = "stablehlo.batch_norm_inference"(%operand, %scale, %offset, %mean, %variance) {
    epsilon = 0.0 : f32,
    feature_index = 2 : i64
  } : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<2x2x2xf64>
  check.expect_eq_const %result, dense<[[[0.0, 0.0], [2.0, 2.0]],
                                        [[2.0, 2.0], [0.0, 0.0]]]> : tensor<2x2x2xf64>
  func.return
}
