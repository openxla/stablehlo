// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[1], [0], [1]]> : tensor<3x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3xf32>, tensor<3xf32>)
    %2 = call @expected() : () -> tensor<3xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<3xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3xf32>, tensor<3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xf32>, tensor<3xf32>) {
    %0 = stablehlo.constant dense<[-1.08914602, 1.31267011, -1.15630043]> : tensor<3xf32>
    %1 = stablehlo.constant dense<[4.31901026, -1.3235271, 1.09718823]> : tensor<3xf32>
    return %0, %1 : tensor<3xf32>, tensor<3xf32>
  }
  func.func private @expected() -> tensor<3xf32> {
    %0 = stablehlo.constant dense<[-1.08914602, 4.31901026, -1.15630043]> : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}

