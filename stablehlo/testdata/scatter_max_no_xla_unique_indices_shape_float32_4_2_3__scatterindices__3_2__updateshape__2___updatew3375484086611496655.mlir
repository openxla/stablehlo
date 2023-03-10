// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[3, 2]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3xf32>, tensor<2xf32>)
    %2 = call @expected() : () -> tensor<4x2x3xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf32>, tensor<2xi32>, tensor<2xf32>) -> tensor<4x2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[[[-6.759640e-01, 3.57921195, -0.649877369], [-1.784554, 0.431616575, 1.65851843]], [[4.80671453, -1.40704131, -2.61496806], [-2.15587831, -0.676147223, 0.884050905]], [[2.09593797, 0.129074484, 1.10057461], [-0.517243683, 2.435080e+00, -1.1138401]], [[-0.524195611, 0.144998312, 1.82032669], [-1.78433263, -0.680296063, 6.23081732]]]> : tensor<4x2x3xf32>
    %1 = stablehlo.constant dense<[2.25946379, 3.61804175]> : tensor<2xf32>
    return %0, %1 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<4x2x3xf32> {
    %0 = stablehlo.constant dense<[[[-6.759640e-01, 3.57921195, -0.649877369], [-1.784554, 0.431616575, 1.65851843]], [[4.80671453, -1.40704131, -2.61496806], [-2.15587831, -0.676147223, 0.884050905]], [[2.09593797, 0.129074484, 1.10057461], [-0.517243683, 2.435080e+00, -1.1138401]], [[-0.524195611, 0.144998312, 2.25946379], [-1.78433263, -0.680296063, 6.23081732]]]> : tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}

