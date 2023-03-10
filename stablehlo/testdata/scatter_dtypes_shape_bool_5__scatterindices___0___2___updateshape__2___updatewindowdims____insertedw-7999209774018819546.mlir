// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5xi1>, tensor<2xi1>)
    %2 = call @expected() : () -> tensor<5xi1>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i1>, %arg1: tensor<i1>):
      stablehlo.return %arg1 : tensor<i1>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>} : (tensor<5xi1>, tensor<2x1xi32>, tensor<2xi1>) -> tensor<5xi1>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5xi1>, tensor<5xi1>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5xi1>, tensor<2xi1>) {
    %0 = stablehlo.constant dense<true> : tensor<5xi1>
    %1 = stablehlo.constant dense<true> : tensor<2xi1>
    return %0, %1 : tensor<5xi1>, tensor<2xi1>
  }
  func.func private @expected() -> tensor<5xi1> {
    %0 = stablehlo.constant dense<true> : tensor<5xi1>
    return %0 : tensor<5xi1>
  }
}

