module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[1], [0], [1]]> : tensor<3x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3xui8>, tensor<3xui8>)
    %2 = call @expected() : () -> tensor<3xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<3xui8>, tensor<3x1xi32>, tensor<3xui8>) -> tensor<3xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3xui8>, tensor<3xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xui8>, tensor<3xui8>) {
    %0 = stablehlo.constant dense<[1, 2, 3]> : tensor<3xui8>
    %1 = stablehlo.constant dense<[0, 0, 1]> : tensor<3xui8>
    return %0, %1 : tensor<3xui8>, tensor<3xui8>
  }
  func.func private @expected() -> tensor<3xui8> {
    %0 = stablehlo.constant dense<[1, 3, 3]> : tensor<3xui8>
    return %0 : tensor<3xui8>
  }
}
