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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x125xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xi8>, tensor<1x125xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xi8>, tensor<1xi8>) {
    %0 = stablehlo.constant dense<"0xFFFEFEFFFFFE01FD0300FD0001FE010003FF010201FD01FD030000FD0000FC05FC00FDFEFEFE0102010004FCFC0200000102050000000001010004FEFD0100010000FC00FE04FE0005FF01FDFC01040100FEFD02000000FE04FEFC0000FEFFFFF9010100000100FEFF000301000002000300000104FF0000FC02050000"> : tensor<1x125xi8>
    %1 = stablehlo.constant dense<2> : tensor<1xi8>
    return %0, %1 : tensor<1x125xi8>, tensor<1xi8>
  }
  func.func private @expected() -> tensor<1x125xi8> {
    %0 = stablehlo.constant dense<"0xFFFEFEFFFFFE01FD0300FD0001FE010003FF010201FD01FD030000FD0000FC05FC00FDFEFEFE0102010004FCFC0200000102050000000001010004FEFD0100010000FC00FE04FE0005FF01FDFC01040100FEFD02000000FE04FEFC0000FEFFFFF9010100000100FEFF000301000002000300000104FF0000FC02050000"> : tensor<1x125xi8>
    return %0 : tensor<1x125xi8>
  }
}

