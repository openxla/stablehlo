// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>)
    %2 = call @expected() : () -> tensor<4x2x3x5xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi8>, tensor<2xi32>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>) {
    %0 = stablehlo.constant dense<"0x00FE0200FD00FE00FF04010000FA0100FE010102FFFF0203FFFD000404010100FFFFFFFCFFFD0000010202020000010102FEFFFE00FF000007FC00FB03FCF80005FF000000FAFD0003FD0300FE00FB01FC000100FE00FEFE0600FEFF0003FF00FFFE00090202FDFFFCFCFF00FB04FFFF00FDFDFF00FDFF00"> : tensor<4x2x3x5xi8>
    %1 = stablehlo.constant dense<[[4, -1, 3], [0, -5, -2], [0, 0, 2], [0, 2, -2]]> : tensor<4x3xi8>
    return %0, %1 : tensor<4x2x3x5xi8>, tensor<4x3xi8>
  }
  func.func private @expected() -> tensor<4x2x3x5xi8> {
    %0 = stablehlo.constant dense<"0x00FE0200F400FE00FFFC010000FA0300FE010102FFFF0203FFFD000404010100FFFF00FCFFFD0000010202020000010102FEFFFE00FF000007FC00FB03FCF80000FF00000000FD0003FD0600FE00FB01FC000100FE00FEFE0600FEFF00030000FFFE00120202FDFF08FCFF00FB04FFFF00FDFDFF00FDFF00"> : tensor<4x2x3x5xi8>
    return %0 : tensor<4x2x3x5xi8>
  }
}

