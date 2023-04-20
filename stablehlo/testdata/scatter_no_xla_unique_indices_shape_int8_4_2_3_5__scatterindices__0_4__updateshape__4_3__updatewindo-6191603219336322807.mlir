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
      stablehlo.return %arg1 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi8>, tensor<2xi32>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>) {
    %0 = stablehlo.constant dense<"0x0102020109FF00FEFF04000300000200FE000200FF02FFFF0204000000FE03FF0100FDFD040300040401F90002FF0300FD0101FE03FFFDFD00000300F9FF0200FCFB0100040006FC060000030000000000FC03000100FFFF00FCFAFF07FC020000FD03FD050202010002FEFA02FFFC01FC03FBFDFB000103"> : tensor<4x2x3x5xi8>
    %1 = stablehlo.constant dense<[[-4, -3, 1], [0, 0, -1], [0, 0, 0], [0, 1, 1]]> : tensor<4x3xi8>
    return %0, %1 : tensor<4x2x3x5xi8>, tensor<4x3xi8>
  }
  func.func private @expected() -> tensor<4x2x3x5xi8> {
    %0 = stablehlo.constant dense<"0x01020201FCFF00FEFFFD000300000100FE000200FF02FFFF0204000000FE03FF010000FD040300000401F900FFFF0300FD0101FE03FFFDFD00000300F9FF020000FB0100040006FC060000030000000000FC03000100FFFF00FCFAFF07FC000000FD0301050202010102FEFA02FFFC01FC03FBFDFB000103"> : tensor<4x2x3x5xi8>
    return %0 : tensor<4x2x3x5xi8>
  }
}

