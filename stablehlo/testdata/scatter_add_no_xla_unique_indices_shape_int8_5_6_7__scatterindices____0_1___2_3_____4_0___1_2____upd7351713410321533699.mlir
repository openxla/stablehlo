// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>)
    %2 = call @expected() : () -> tensor<5x6x7xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x2xi32>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>) {
    %0 = stablehlo.constant dense<"0x00FF0100020105FE010500030003010301FDFDFA040000FB040000010000010005FFFF020001FE00FC000201FF000102FF00FEFF0406FFFEFF000400010400FE0400FD00FB02FF00FE0001FF0405FB030303FF00F9FF04FDFFFF060000FF0504000004FE08FB00FF02020000020400FD01FE01FF0803FE0100FF00FFFC02FDFF00FFFE010101FBFDFE02FFFF00FE01000200000102000000000101FE01000200FE01FAFF0102FD0200F906FB01FF0002FF0203FD00FFFE00050101000100000001FEFEFFFE00000003000101FC0202FD02FE"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[[3, 2], [2, 0]], [[4, 0], [-2, -3]], [[-4, 0], [0, 0]], [[0, 0], [4, 0]], [[0, -3], [0, 0]]]> : tensor<5x2x2xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x00020100020105FE010500030003010301FFFDFA040000FB040000010200010005FFFF020001FE00FC000205FF000102FF00FEFC0406FFFEFF000400010400FE0400FD00FB02FD00FE0001FF0405FB030303FF00F9FB04FDFFFF060000FF0504000004FE08FB00FF02020000020400FD01FE01FF0803FE0100FF00FFFC02FDFF00FFFE010101FBFDFE02FFFF00FE01000200000102000000000105FE01000200FE01FAFF0102FD0200F906FB01FF0002FF0203FD00FFFE0005FE01000100000001FEFEFFFE00000003000101FC0202FD02FE"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

