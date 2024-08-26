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
      %3 = stablehlo.add %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<5x6x7xi8>, tensor<2x2x2xi64>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    return %2 : tensor<5x6x7xi8>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}, tensor<5x2x2xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFF03030003050001FF0101FD0301020100FF000000FF0205FC000204FF0500010102FC000007050202050103020004040000FEFFFEFFFF000203FB01FDFFFE0000FF040102020200010000020003000304FEFCFC08FEFF00FCFEFE0100FC010106FE03000003FE0004FDFB00FEFC0105050202FD000202FDFC02010004FEFDFC02020200FEFFFFFE020300FF00010304000500FD050400FFFD00FC0104FBFE00050606FFFD01FD00FCFF0101FE00FE0100F904FE0203FD0200FFFD00000002000400FEFE04000002FE02FF0301FFFF020003"> : tensor<5x6x7xi8>
    %c_0 = stablehlo.constant dense<[[[-1, -3], [0, 0]], [[0, 2], [1, -4]], [[-3, -2], [-1, -1]], [[1, 2], [1, -3]], [[0, 1], [-1, 3]]]> : tensor<5x2x2xi8>
    return %c, %c_0 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFF02030003050001FF0101FD0301020100FC000000FF0205FC000204FF0500010102FC000007050202050103020004040000FEFBFEFFFF000203FB03FDFFFE0000FF040102020300010000020003000304FEFCFC08FBFF00FCFEFE0100FB010106FE03000001FE0004FDFB00FEFC0105040202FD000202FDFC02010004FEFDFD02020200FEFFFFFB020300FF00010306000500FD050400FFFD00FD0104FBFE00050606FFFD01FD00FCFF0101FE00FE0100FC04FE0203FD020000FD00000002000400FEFE03000002FE02FF0301FFFF020003"> : tensor<5x6x7xi8>
    return %c : tensor<5x6x7xi8>
  }
}
