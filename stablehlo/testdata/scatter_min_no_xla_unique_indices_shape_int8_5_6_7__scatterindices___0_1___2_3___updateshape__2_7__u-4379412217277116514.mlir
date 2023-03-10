// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<2x7xi8>)
    %2 = call @expected() : () -> tensor<5x6x7xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2xi32>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<2x7xi8>) {
    %0 = stablehlo.constant dense<"0x00FF01FDFE06FF00FEFC00FAFF01FFFFFE01010000010106000000FE010004FC040200FE00FEFD040401FE00FE0201020001010000F903000203FFFDFE02FD0001000104010300040102030000FD00FEFD0303FF0002060201FD0203FCF9FFFC03FD05FEFCFC03050204FE0003FEFF00FBFFFF0000FD00FF040004FFFDFFFEFEFE00FCFF00FE01FF0600000605030100FDFB00000300FD0100FDFB05FF0004FFFBFA0000FD02FF00FFFD000001FCFE03FD07FD00FE0400FDFE0001000303FA00FD02010200010000FE040102FCFAFA010500"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[-2, -2, 2, 0, 0, 2, 0], [1, 0, -3, -5, -1, 0, 0]]> : tensor<2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x00FF01FDFE06FFFEFEFC00FAFF00FFFFFE01010000010106000000FE010004FC040200FE00FEFD040401FE00FE0201020001010000F903000203FFFDFE02FD0001000104010300040102030000FD00FEFD0303FF0002060201FD0203FCF9FFFC03FD05FEFCFC03050201FEFDFBFEFF00FBFFFF0000FD00FF040004FFFDFFFEFEFE00FCFF00FE01FF0600000605030100FDFB00000300FD0100FDFB05FF0004FFFBFA0000FD02FF00FFFD000001FCFE03FD07FD00FE0400FDFE0001000303FA00FD02010200010000FE040102FCFAFA010500"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

