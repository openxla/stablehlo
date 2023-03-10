// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xbf16>, tensor<1xbf16>)
    %2 = call @expected() : () -> tensor<1x125xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xbf16>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1x125xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xbf16>, tensor<1xbf16>) {
    %0 = stablehlo.constant dense<"0x4ABF4FBF883F2CC008C0C5C0C6400CC0EDBED9C0914031C08E402DC00240ABBFFE3E9A409E40DDBF21C01540683EF73F6E3F59C0A7C00B4023C0B93F114032BFA13F3FBF6FC0BABF44C0B0BFF9BF96BF2B3F10BFECBF3EC01EC093408EC0CF3F4C402D409B40893CB6BF0B4074C022C0DDBFBDBFB6C01D3F7F40B4BE6CC06CBEB53FBBBFEB3F09403BC08D406C404B40A3BE2D406B408A40DABED3BEABBFB3BF76403EBE2C3FB9C0453FD9BF19C00EC040C031408A4018C04A3F4840B2C0044001C0DE3E86C0D2C0FE3F0DC021C01140743F104012402540EDBF59BF0D402940394015BF8AC0DCC097409C4061400B40A33FF23FE1BF3A40AEC0"> : tensor<1x125xbf16>
    %1 = stablehlo.constant dense<1.348880e-02> : tensor<1xbf16>
    return %0, %1 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> tensor<1x125xbf16> {
    %0 = stablehlo.constant dense<"0x4ABF4FBF883F2CC008C0C5C0C6400CC0EDBED9C0914031C08E402DC00240ABBFFE3E9A409E40DDBF21C01540683EF73F6E3F59C0A7C00B4023C0B93F114032BFA13F3FBF6FC0BABF44C0B0BFF9BF96BF2B3F10BFECBF3EC01EC093408EC0CF3F4C402D409B40893CB6BF0B4074C022C0DDBFBDBFB6C01D3F7F40B4BE6CC06CBEB53FBBBFEB3F09403BC08D406C404B40A3BE2D406B408A40DABED3BEABBFB3BF76403EBE2C3FB9C0453FD9BF19C00EC040C031408A4018C04A3F4840B2C0044001C0DE3E86C0D2C0FE3F0DC021C01140743F104012402540EDBF59BF0D402940394015BF8AC0DCC097409C4061400B40A33FF23FE1BF3A40AEC0"> : tensor<1x125xbf16>
    return %0 : tensor<1x125xbf16>
  }
}

