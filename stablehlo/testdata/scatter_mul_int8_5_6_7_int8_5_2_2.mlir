// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>)
    %1 = call @expected() : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<5x6x7xi8>, tensor<2x2x2xi64>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    return %2 : tensor<5x6x7xi8>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}, tensor<5x2x2xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x020300000401FBFBFEFD000101FF0000FE01FFFC00FFFD01FBFE02FD02FD0601FE00030100FA0000FF060102FD010500FEFE02000200F8FF00010001FF0002030200040000000000000000000004020002000102FFFF0301FC00F6FF030200FF03FE01010004000001FE00000002FFFE0400FE00FFFD02FFFE0000F8FF010002FEFCFEFE0400FDFF03FD000000030600020100FF02FEFF040000FE00010201FE01FF02000205FC0200FD0701FE0200FF00FF0200FD04000200FE00F8010001010001FFFE000000FCFF02020302FEFFFFFF00"> : tensor<5x6x7xi8>
    %c_0 = stablehlo.constant dense<[[[3, -5], [1, 1]], [[-1, -6], [0, 5]], [[-1, 2], [-1, -3]], [[-1, 1], [-3, -2]], [[-2, -1], [4, -1]]]> : tensor<5x2x2xi8>
    return %c, %c_0 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x020900000401FBFBFEFD000101FF0000FEFBFFFC00FFFD01FBFE02FD02FD0601FE00030100FA0000FF0601FEFD010500FEFE02000200F8FF000100FAFF0002030200040000000000000000000004020002000102FF010301FC00F6FF03FA00FF03FE01010008000001FE00000002FFFEFC00FE00FFFD02FFFE0000F8FF0100FEFEFCFEFE0400FD0203FD000000030600020100FF02FEFF0400000600010201FE01FF02000205FC0200060701FE0200FF00010200FD040002000200F8010001010001FFFE000000FCFF02020302FEFFFFFF00"> : tensor<5x6x7xi8>
    return %c : tensor<5x6x7xi8>
  }
}
