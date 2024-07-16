// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<2x7xi8>)
    %1 = call @expected() : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      stablehlo.return %arg1 : tensor<i8>
    }) : (tensor<5x6x7xi8>, tensor<2x2xi64>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    return %2 : tensor<5x6x7xi8>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}, tensor<2x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFD0403FE00FB020002FC0100FD00FEFF0104FF0001FEFD00000100FF01020001020101040202FE02FB0100FEFD010106FB0201FF00000302FC0300FFFF00000101FEFFFC01FFFDFF00FC04FE0300030000FFFEFF00000204FE0301FD020102FEFF020201000701FE02030603FDFAFFFBFBFD00FE04000100010400FFF802FC03FEFFFEFB00FE050000FCFAFFFF00FF0002FE00FE020000FF05050000000300030002000101FDFD000005020001FB02FFFC0002FF040200FD00FCFF0000FFFD07FF000400FF0000020600030102FF030005FD"> : tensor<5x6x7xi8>
    %c_0 = stablehlo.constant dense<[[1, 1, 2, -3, 1, 6, -2], [-4, 0, 1, 0, 3, 0, 1]]> : tensor<2x7xi8>
    return %c, %c_0 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFD0403FE00FB02010102FD0106FEFEFF0104FF0001FEFD00000100FF01020001020101040202FE02FB0100FEFD010106FB0201FF00000302FC0300FFFF00000101FEFFFC01FFFDFF00FC04FE0300030000FFFEFF00000204FE0301FD020102FEFF020201000701FE02FC000100030001FBFD00FE04000100010400FFF802FC03FEFFFEFB00FE050000FCFAFFFF00FF0002FE00FE020000FF05050000000300030002000101FDFD000005020001FB02FFFC0002FF040200FD00FCFF0000FFFD07FF000400FF0000020600030102FF030005FD"> : tensor<5x6x7xi8>
    return %c : tensor<5x6x7xi8>
  }
}
