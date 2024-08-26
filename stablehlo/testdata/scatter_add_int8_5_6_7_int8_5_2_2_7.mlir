// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>)
    %1 = call @expected() : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<5x6x7xi8>, tensor<2x2x1xi64>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    return %2 : tensor<5x6x7xi8>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}, tensor<5x2x2x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0000FF00000402FE00FF0004000000000303FE00FDFC01FD0101020105FB00FA0204FC0004FF010000FFFBFFFE020001FF00FC0003000003000203FDFF0303FEFFFEFFFF0001000401FFFCFEFE02020004FF0000FF01000001FD0200FF06FF01FDFDFD010301030005020000FFFE01FCFD0402000103FFFC02FFFC05FF0203FFFB03FF0005FFFD0007FEFEFF00000101FD00000103FDFEFEFFFB03FEFEFC03FE01FDFFFEFE0101FD02FF05FDFF020000FB0300FBFD02FB050000FAFF010301FFFF0501FD0400FF0701FCFEFF000002FF03FF"> : tensor<5x6x7xi8>
    %c_0 = stablehlo.constant dense<"0xFD0500FEF9F8000100000100FC00FE070001FE010300020304FC01FF0003FDFD0100FF0000000000FD000404000003FCFD04000201FE0000010002FF0303020102FC01020003FFFE08FF00FE00FB00FD0300FF03FF0101FFFE01030002FD00FE0002FD000000020005FEFCFCFF02FD0000FFFF0600FE00FFFDFF030404FB00FF00FD0001FBFF0000F901FE00"> : tensor<5x2x2x7xi8>
    return %c, %c_0 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFD05FFFEF9FC02FF00FF0104FC00FE070304FC0100FC030005FD030005FB00FA0204FC0004FF010000FFFB02FBFF0101FE00FC000300FD03040603FD02FF0002FF0000FD0001000401FFFCFEFE02020004FF0000000102FF0400040101020003FD00FCFF0B0003FE05FD00FD02FE00FFFD0402000103FFFC02FFFC05FF020200FC02FD0108FFFFFD07FCFE01FD000101FF0005FFFFF9FD00FCFB03FEFEFC03FE01FDFFFEFE0101FD02FE0403FF0000FFF80203FF01FDFB0400FDFA00FC0201FFF806FFFD0400FF0701FCFEFF000002FF03FF"> : tensor<5x6x7xi8>
    return %c : tensor<5x6x7xi8>
  }
}
