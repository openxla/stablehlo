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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<5x6x7xi8>, tensor<2x2xi64>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    return %2 : tensor<5x6x7xi8>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}, tensor<2x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFEFDFC0006FE03020300FF00FE01FE00FFFD000103FDFFFC000604FF00FE0204000403FE01010604FE00FC0300FF000300FEFC00FD02FD00010402FA02FB0200000100FFFF08FD0204020204FDFE040101FD00060008FC00000501FEFE030001FB03FDFD0304FD03000000010002000000000300FF0000FF03000408FFFF00FF0200F702030002040100FCFF0403060100FDFF0005FF01FFFE000406FF040100FCFDFDFB02FE02000200FD030200000000FC00FFFD000302FEFBFFFD030300FFFFFE0100FFFF0403FD0000FCFBFC00FE0503"> : tensor<5x6x7xi8>
    %c_0 = stablehlo.constant dense<[[0, 4, -3, 4, 2, -3, 0], [1, 1, 1, -3, -3, 6, 2]]> : tensor<2x7xi8>
    return %c, %c_0 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFEFDFC0006FE030003FDFF00FD00FE00FFFD000103FDFFFC000604FF00FE0204000403FE01010604FE00FC0300FF000300FEFC00FD02FD00010402FA02FB0200000100FFFF08FD0204020204FDFE040101FD00060008FC00000501FEFE030001FB03FDFD0304FD0300000001FDFD000000000300FF0000FF03000408FFFF00FF0200F702030002040100FCFF0403060100FDFF0005FF01FFFE000406FF040100FCFDFDFB02FE02000200FD030200000000FC00FFFD000302FEFBFFFD030300FFFFFE0100FFFF0403FD0000FCFBFC00FE0503"> : tensor<5x6x7xi8>
    return %c : tensor<5x6x7xi8>
  }
}
