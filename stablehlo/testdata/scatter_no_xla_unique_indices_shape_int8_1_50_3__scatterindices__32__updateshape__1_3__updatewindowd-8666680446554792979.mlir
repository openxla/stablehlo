// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xi8>, tensor<1x3xi8>)
    %2 = call @expected() : () -> tensor<1x50x3xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      stablehlo.return %arg1 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi8>, tensor<1xi32>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi8>, tensor<1x3xi8>) {
    %0 = stablehlo.constant dense<"0x0801FF00FB02FF00000201FF01FF01FE0002FE0000FF0301FFFF0004010200FC00FEFB03010300FC0700FFFD020100FE02FEFA00FF03FDF90200FD00FBFFFF02010301FF00000003000200FF03FFFF0300FD00FE0300000304030006040500FF02FEFB0300FF03FEFFFE04FD03FF0007000101FDFE0300FF00020302040001FC01FF000000FEFC00FEFFFFFF030003FE010101FC00FE"> : tensor<1x50x3xi8>
    %1 = stablehlo.constant dense<[[-1, -3, 1]]> : tensor<1x3xi8>
    return %0, %1 : tensor<1x50x3xi8>, tensor<1x3xi8>
  }
  func.func private @expected() -> tensor<1x50x3xi8> {
    %0 = stablehlo.constant dense<"0x0801FF00FB02FF00000201FF01FF01FE0002FE0000FF0301FFFF0004010200FC00FEFB03010300FC0700FFFD020100FE02FEFA00FF03FDF90200FD00FBFFFF02010301FF00000003000200FF03FFFF0300FD00FE0300000304030006040500FFFFFD010300FF03FEFFFE04FD03FF0007000101FDFE0300FF00020302040001FC01FF000000FEFC00FEFFFFFF030003FE010101FC00FE"> : tensor<1x50x3xi8>
    return %0 : tensor<1x50x3xi8>
  }
}

