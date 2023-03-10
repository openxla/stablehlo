// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5xf32>, tensor<2xf32>)
    %2 = call @expected() : () -> tensor<5xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<5xf32>, tensor<2x1xi32>, tensor<2xf32>) -> tensor<5xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5xf32>, tensor<5xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[-5.54396296, -5.21857262, -2.2443738, -0.273012817, -3.77916503]> : tensor<5xf32>
    %1 = stablehlo.constant dense<[-3.52032351, -2.88679481]> : tensor<2xf32>
    return %0, %1 : tensor<5xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<5xf32> {
    %0 = stablehlo.constant dense<[-9.06428623, -5.21857262, -5.13116837, -0.273012817, -3.77916503]> : tensor<5xf32>
    return %0 : tensor<5xf32>
  }
}

