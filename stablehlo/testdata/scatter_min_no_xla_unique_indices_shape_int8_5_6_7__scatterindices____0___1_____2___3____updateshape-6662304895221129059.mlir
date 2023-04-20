// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>)
    %2 = call @expected() : () -> tensor<5x6x7xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x1xi32>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>) {
    %0 = stablehlo.constant dense<"0x0100FF00FFFE00FD00FD00FEFFFF0300FE040003FC0102FCFBFD000000FB00FE0100020003FC01FD010200FF010005000102FFFCFDFEFDFD000003FFFFFDFB010300040001040201FF0100FFFE030803FC04FD02FE010302050100020002FDFD010003FDFFFD000202040102FE0003FFFFFC00FFFF0100020203FC0102FF03FEFCFC01FD000004000008FC00FEFF010500FE00FFFE02FD0100FE040000010002000501FFFF00FB080201FF03020105FFFF00FF04FFFD0000010100FFFA04FE000103FFFE0100FFFE0004FC02FDFC000301FE"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<"0xF901050402010000FEFE00FB00FFFE080102050102FCFEF800FFFD00FC0104FD0305FDFE0000FD02000000FF0203FF01FEFEFDFFFC0201FC0201FA0400010500FFFB0000FFFFFD0300050201FD0004FDFFFFFC02FF0102FCFF0303FCFC0402FDFE03FEFD01FE0200FFFF01FEFF0005FEFC030200FE0203FDFF0300FF04FDFE00F90001030100FF0005FB0800"> : tensor<5x2x2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0xF900FF00FFFE00FDFEFD00FBFFFFFE00FE020001FCFCFEF8FBFDFD0000FB00FE0100020003FC01FD0102FCFF01FD0300FDFEFFFCFDFEFDFD00FF02FFFFFDFBFEFDFFFC0001FC0201FF0100FFFE030803FC04FD02FE01FA0200010000FFFBFDFDFFFFFDFDFFFD0001FD0001FDFEFFFCFFFFFC00FFFF0100020203FC0102FFFFFEFCFCFFFD00FCFC0000FDFC00FEFD01FE00FEFFFFFEFEFD0000FE040000010002000501FFFF00FB08FC01FF00FE0103FDFF00FFFFFFFDFE00F90000FFFA00FE0001FBFFFE0100FFFE0004FC02FDFC000301FE"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

