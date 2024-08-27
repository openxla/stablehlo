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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<5x6x7xi8>, tensor<2x2x1xi64>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    return %2 : tensor<5x6x7xi8>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}, tensor<5x2x2x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00FCFFFE00FC010000000100FF01FD00000002FF01020000FE00FF00FD00FF0100030002030001FEFF010500FFFE01FD01FEFD0002FCFD00FFFEFD01000000FB02FF01FD06FC07FF00000101FF0003FCFEFE030204FF01FDFD0202FF000802020000FF04FAFA000200FEFF03FBFD040000FFFC000200FB0404000100FB0201FDFCFDFF000105FF0101FF00FE02FE0205000002FD00FFFBFEFF01020200FEFE02000000000003000000FFFBFD00FF00FF0000000000070402FCFC000000FEFFFE00000002000002030202FEFF05FE000500FF"> : tensor<5x6x7xi8>
    %c_0 = stablehlo.constant dense<"0xFFFE00FF00FE05FF0206FC00FC00010502FEFF0201FE00FE0000FC0503FFFB03FDFC04FEFCFC01030101FB04030601010000FEFE0001FF0001FE06FD0002060000030000FF0003FA0301FA0002FEFF000202FDFEFFF9FAF9FEFE0209FFFDFD0102FC0004010100FA05FFFB06FBFE000305FFFDFE0202000404FFFF010202FF0401F800FE050103FCFDFEFF00"> : tensor<5x2x2x7xi8>
    return %c, %c_0 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFFFCFFFE00FC01FF0000FC00FC00FD0000FEFFFF01FE00FEFE00FC00FD00FF0100030002030001FEFF0103FFFBFEFDFC01FEFCFC01FCFD00FBFEFD01000000FBFEFE00FDFFFC07FF00000101FF0003FCFEFE030201FE01FDFD0202FF00030000FF00FFFAFAFAFA0000FEFF00FBFDFDFE00FFFC000200FB0404000100FB02FFF9FAF9FEFE0105FFFDFDFF00FC00FE010100FA02FDFBFFFBFEFF01020200FEFE02000000000003000000FFFBFD00FF00FF00FFFF000002FF02FCF800FE00FEFFFCFDFEFF00000002030202FEFF05FE000500FF"> : tensor<5x6x7xi8>
    return %c : tensor<5x6x7xi8>
  }
}
