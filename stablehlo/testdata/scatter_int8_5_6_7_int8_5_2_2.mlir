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
      stablehlo.return %arg1 : tensor<i8>
    }) : (tensor<5x6x7xi8>, tensor<2x2x2xi64>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    return %2 : tensor<5x6x7xi8>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}, tensor<5x2x2xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00FF01FD00000001FFFCFFFB0300FF0103000004FF000601030000FBFE020005FC000200FF0000FF0300FEFD0109000203FFFC00FE01FEFDFE00FA01FD000905FD00FF0305FEFF01FE010204000002FE0500FF03FE02FB0003000003010000020302000100000502FC0705000100FEFFFF0101FCFD0703010006FA0100FCF800FF0006FD00020500FBFE00FB0001FD01070100FEFE0000FFFE020202FD01FC0300000002020401FE00FA01FE0204030401FEFF00FCFF0400FFFFFA0100020000000402FDFC00FBFCFCFE000200FE01010500"> : tensor<5x6x7xi8>
    %c_0 = stablehlo.constant dense<[[[0, 0], [0, 2]], [[2, 7], [-4, 2]], [[0, 0], [4, -5]], [[0, 5], [0, -1]], [[0, -5], [-1, -1]]]> : tensor<5x2x2xi8>
    return %c, %c_0 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000001FD00000001FF02FFFB0300FF0103000004FF000601030000FB00020005FC000200FF0000FF0300FE020109000203FFFC02FE01FEFDFE00FA07FD000905FD00FF0305FEFC01FE010204000002FE0500FF03FE00FB000300000301FB00020302000100000502FC0705000100FEFF040101FCFD0703010006FA0100FCF800FF0006FD000205FFFBFE00FB0001FD05070100FEFE0000FFFE020002FD01FC0300000002020401FE000001FE0204030401FFFF00FCFF0400FFFBFA0100020000000402FDFF00FBFCFCFE000200FE01010500"> : tensor<5x6x7xi8>
    return %c : tensor<5x6x7xi8>
  }
}
