// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[3, 2]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3xui8>, tensor<2xui8>)
    %2 = call @expected() : () -> tensor<4x2x3xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xui8>, tensor<2xi32>, tensor<2xui8>) -> tensor<4x2x3xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xui8>, tensor<4x2x3xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xui8>, tensor<2xui8>) {
    %0 = stablehlo.constant dense<[[[1, 4, 6], [3, 4, 9]], [[0, 4, 6], [3, 1, 1]], [[0, 2, 5], [0, 0, 0]], [[1, 0, 0], [4, 0, 0]]]> : tensor<4x2x3xui8>
    %1 = stablehlo.constant dense<[5, 6]> : tensor<2xui8>
    return %0, %1 : tensor<4x2x3xui8>, tensor<2xui8>
  }
  func.func private @expected() -> tensor<4x2x3xui8> {
    %0 = stablehlo.constant dense<[[[1, 4, 6], [3, 4, 9]], [[0, 4, 6], [3, 1, 1]], [[0, 2, 5], [0, 0, 0]], [[1, 0, 5], [4, 0, 6]]]> : tensor<4x2x3xui8>
    return %0 : tensor<4x2x3xui8>
  }
}

