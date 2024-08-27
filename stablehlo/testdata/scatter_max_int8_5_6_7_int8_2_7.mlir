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
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<5x6x7xi8>, tensor<2x2xi64>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    return %2 : tensor<5x6x7xi8>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}, tensor<2x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFE00FF0000FE0201FF01000000FF00FD020001040200FDFF07010800FDFF0503FD0200FA01FDFF000003FEFAFE0101040201FDFF00FAFF0000FAFE03FDFCFC0102FE02FF050003FD01FF020003FE00FEFD010002FB0304020401FF0403FEFEFFFA040002FEFD010302FB00FF04FFFB0001FE0000FAFDFF010403040100000204FFFFFFFDFF01FE0200FAFF04000000FDFE00FFF9FD0200000200FEFD01000000FFFD01FF01FD00FF00FFFEFDFE0200FFFF00FD05FFFDFD000000FE01FE01000600FEFE04FF0600020402FDFE010200F90000"> : tensor<5x6x7xi8>
    %c_0 = stablehlo.constant dense<[[0, 0, 2, -1, -5, -1, 4], [1, 3, -3, 3, 4, 0, 0]]> : tensor<2x7xi8>
    return %c, %c_0 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFE00FF0000FE020100020000000400FD020001040200FDFF07010800FDFF0503FD0200FA01FDFF000003FEFAFE0101040201FDFF00FAFF0000FAFE03FDFCFC0102FE02FF050003FD01FF020003FE00FEFD010002FB0304020401FF0403FEFEFFFA040002FEFD0103020103FF0404000001FE0000FAFDFF010403040100000204FFFFFFFDFF01FE0200FAFF04000000FDFE00FFF9FD0200000200FEFD01000000FFFD01FF01FD00FF00FFFEFDFE0200FFFF00FD05FFFDFD000000FE01FE01000600FEFE04FF0600020402FDFE010200F90000"> : tensor<5x6x7xi8>
    return %c : tensor<5x6x7xi8>
  }
}
