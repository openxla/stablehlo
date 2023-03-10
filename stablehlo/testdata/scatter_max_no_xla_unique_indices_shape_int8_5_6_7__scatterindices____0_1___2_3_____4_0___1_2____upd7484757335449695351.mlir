// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>)
    %2 = call @expected() : () -> tensor<5x6x7xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x2xi32>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>) {
    %0 = stablehlo.constant dense<"0x04FE0003FCFF04020305010004000101010001000403000104FE000501FE0302FFFE00FEFFFD0007FEFD00FF01FF0200FCFF00FDF7FF03FE0201030400FFFFFDFAFE03FF04020000FFFFFEFD00FC0002010003000102050401FE000603000200FD00FD010102FCFF00000100040002030304FCFDFF000000FF05F60503FD00FA00FD01000402040301FFFE00030200FC01FFFC02FFFD00FCFBFEFBFE0005FE000206010500FE010002FF0003020200010000FD04FAFD0200000001000202F8040600FF0000FEFD0200FD00FE050100000401"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[[-2, 1], [-1, 3]], [[-4, -1], [-2, -3]], [[3, -4], [-1, -1]], [[-1, 2], [-1, 4]], [[0, -5], [-2, -7]]]> : tensor<5x2x2xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x04FE0003FCFF04020305010004000101010101000403000104FE000501FE0302FFFE00FEFFFD0007FEFD00FF01FF0200FCFF00FDF7FF03FE0201030400FFFFFDFAFE03FF04020000FFFFFEFD00FC0002010003000103050401FE000603000200FD00FD010102FCFF00000100040002030304FCFDFF000000FF05F60503FD00FF00FD01000402040401FFFE000302000201FFFC02FFFD00FCFBFEFFFE0005FE000206010500FE010002000003020200010000FD04FAFD0200000001000202F8040600FF0000FEFD0200FD00FE050100000401"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

