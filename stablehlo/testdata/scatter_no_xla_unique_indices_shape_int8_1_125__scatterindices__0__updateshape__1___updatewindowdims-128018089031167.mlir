// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xi8>, tensor<1xi8>)
    %2 = call @expected() : () -> tensor<1x125xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      stablehlo.return %arg1 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x125xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xi8>, tensor<1x125xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xi8>, tensor<1xi8>) {
    %0 = stablehlo.constant dense<"0x00FFFEFD000004010204FF00FFFF01FD05FE0201FD00030400010001FF03FC0000020200FD020003050000FDFC00FDFC05FD00FE03FFFF02FF0201FC0100FF0500000106FE00FE02FF03FEFF00FF03FE080101010003FF05FF0201030001FD03FC020401FEFF000003FFFF0200FDFF04FF00FFFE00FE08FF0101FC0000"> : tensor<1x125xi8>
    %1 = stablehlo.constant dense<0> : tensor<1xi8>
    return %0, %1 : tensor<1x125xi8>, tensor<1xi8>
  }
  func.func private @expected() -> tensor<1x125xi8> {
    %0 = stablehlo.constant dense<"0x00FFFEFD000004010204FF00FFFF01FD05FE0201FD00030400010001FF03FC0000020200FD020003050000FDFC00FDFC05FD00FE03FFFF02FF0201FC0100FF0500000106FE00FE02FF03FEFF00FF03FE080101010003FF05FF0201030001FD03FC020401FEFF000003FFFF0200FDFF04FF00FFFE00FE08FF0101FC0000"> : tensor<1x125xi8>
    return %0 : tensor<1x125xi8>
  }
}

