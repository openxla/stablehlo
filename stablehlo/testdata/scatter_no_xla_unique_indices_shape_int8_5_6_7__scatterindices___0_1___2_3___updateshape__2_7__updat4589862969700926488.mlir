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
      stablehlo.return %arg1 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2xi32>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<2x7xi8>) {
    %0 = stablehlo.constant dense<"0x000500FE010001FD00FDFFFDFCFF0104FF02010600FD00040000FD05FEFF00000000FFFF0300FD03FFFFFFFE040504FD06FE04FA00000001020004FF0001FFFF0106FEFB0807FEFE0000FFFFFFFEFDFD01000505FEFF0000000102FF050003000005FF0004FB0000020300040402FD01FAFF01FF01FD030100000900020003F904000003000102020200000002FD04FFFDFD0300FE000201FA00FFFBFF00F901FFFB0300FD0001FBFEFF00FD0307FAFF0201FDFEFE0504FFFF0201F8FFFF04FA01FD02FDFF0001FE000000FF0000FFFC0003"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[0, 0, 1, 0, 2, -2, 0], [-5, 0, -2, 1, -4, -1, -3]]> : tensor<2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x000500FE0100010000010002FE000104FF02010600FD00040000FD05FEFF00000000FFFF0300FD03FFFFFFFE040504FD06FE04FA00000001020004FF0001FFFF0106FEFB0807FEFE0000FFFFFFFEFDFD01000505FEFF0000000102FF050003000005FF0004FB000002FB00FE01FCFFFDFAFF01FF01FD030100000900020003F904000003000102020200000002FD04FFFDFD0300FE000201FA00FFFBFF00F901FFFB0300FD0001FBFEFF00FD0307FAFF0201FDFEFE0504FFFF0201F8FFFF04FA01FD02FDFF0001FE000000FF0000FFFC0003"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

