// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x4xi16>, tensor<3x2x4xi16>)
    %2 = call @expected() : () -> tensor<3x5x4xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xi16>, tensor<2x1xi32>, tensor<3x2x4xi16>) -> tensor<3x5x4xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xi16>, tensor<3x5x4xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xi16>, tensor<3x2x4xi16>) {
    %0 = stablehlo.constant dense<[[[1, -2, -2, -2], [0, 0, 1, 0], [0, -4, -5, -1], [-1, 4, 0, 3], [3, -2, 2, 0]], [[-2, -4, 6, 2], [0, 4, -1, -2], [0, -4, -2, -1], [-1, 3, 0, 0], [1, -1, -1, 0]], [[0, 3, -2, 2], [-1, -2, 1, -4], [-7, -2, 2, 6], [0, -1, 2, 5], [-2, 3, 1, 0]]]> : tensor<3x5x4xi16>
    %1 = stablehlo.constant dense<[[[1, 0, -5, 2], [-3, -4, 0, -3]], [[0, 3, -4, 5], [3, -1, 0, -3]], [[2, -1, -2, -1], [-3, 0, 0, 0]]]> : tensor<3x2x4xi16>
    return %0, %1 : tensor<3x5x4xi16>, tensor<3x2x4xi16>
  }
  func.func private @expected() -> tensor<3x5x4xi16> {
    %0 = stablehlo.constant dense<[[[1, -2, -2, -2], [0, 0, 0, 0], [0, -4, -5, -1], [-1, 4, 0, 3], [3, -2, 2, 0]], [[-2, -4, 6, 2], [0, -12, 0, 30], [0, -4, -2, -1], [-1, 3, 0, 0], [1, -1, -1, 0]], [[0, 3, -2, 2], [6, 0, 0, 0], [-7, -2, 2, 6], [0, -1, 2, 5], [-2, 3, 1, 0]]]> : tensor<3x5x4xi16>
    return %0 : tensor<3x5x4xi16>
  }
}

