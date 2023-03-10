// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<2> : tensor<1x3x1xi32>
    %1:2 = call @inputs() : () -> (tensor<2x3xi8>, tensor<2x1x3xi8>)
    %2 = call @expected() : () -> tensor<2x3xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>} : (tensor<2x3xi8>, tensor<1x3x1xi32>, tensor<2x1x3xi8>) -> tensor<2x3xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<2x3xi8>, tensor<2x3xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi8>, tensor<2x1x3xi8>) {
    %0 = stablehlo.constant dense<[[1, 7, 2], [6, 1, 0]]> : tensor<2x3xi8>
    %1 = stablehlo.constant dense<[[[0, -3, 0]], [[2, -4, 0]]]> : tensor<2x1x3xi8>
    return %0, %1 : tensor<2x3xi8>, tensor<2x1x3xi8>
  }
  func.func private @expected() -> tensor<2x3xi8> {
    %0 = stablehlo.constant dense<[[1, 7, -3], [6, 1, -4]]> : tensor<2x3xi8>
    return %0 : tensor<2x3xi8>
  }
}

