// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @optimization_barrier_op_test() {
  %operand0 = stablehlo.constant dense<0.0> : tensor<f32>
  %operand1 = stablehlo.constant dense<1.0> : tensor<f32>
  %result0, %result1 = stablehlo.optimization_barrier %operand0, %operand1 : tensor<f32>, tensor<f32>
  check.expect_almost_eq_const %result0, dense<0.0> : tensor<f32>
  check.expect_almost_eq_const %result1, dense<1.0> : tensor<f32>
  func.return
}
