// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @batch_norm_grad() {
  %operand = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]],
                                       [[3.0, 4.0], [1.0, 2.0]]]> : tensor<2x2x2xf64>
  %scale = stablehlo.constant dense<[1.0, 1.0]> : tensor<2xf64>
  %mean = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf64>
  %variance = stablehlo.constant dense<[1.0, 1.0]> : tensor<2xf64>
  %grad_output = stablehlo.constant dense<0.1> : tensor<2x2x2xf64>
  %grad_operand, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(%operand, %scale, %mean, %variance, %grad_output) {
    epsilon = 0.0 : f32,
    feature_index = 2 : i64
  } : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>,
       tensor<2x2x2xf64>) -> (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>)
  check.expect_almost_eq_const %grad_operand, dense<0.0> : tensor<2x2x2xf64>
  check.expect_almost_eq_const %grad_scale, dense<[0.0, 0.0]> : tensor<2xf64>
  check.expect_almost_eq_const %grad_offset, dense<[0.4, 0.4]> : tensor<2xf64>
  func.return
}
