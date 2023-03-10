// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>)
    %2 = call @expected() : () -> tensor<1x50x3xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xbf16>, tensor<1xi32>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>) {
    %0 = stablehlo.constant dense<"0x96BED5C00DC01A40BEC03C409BBFD1BFBA3EE8400C408EBE1CC005C01BBECABF253F9440E3BD813F23C08940B83F8440244091BC03C0A8BEDE3D51BF94C083C08CBFF74024C00B40CEBE343E73BFB8BF8B40C8BF2CC0914073C0A6C0CE3F39BF80406AC06ABFF0C082C01BC0FDC083C0E73E653FB93F9E3FB340593FA03E373FABC03140B23FC74041402AC082C0CC3F88C09A402E4160BF1C4018C1C0BF93403540EC3D5040E63E08401BBF8F4090C0733F503F953E76C0733FCD3F1BC013BFB8BF613FFBBF9FBD07C01FC015402340134062BFB8BFACC06B3FE53F9D3E73C00FBE40407CC0B74093BF03BFFCBFAF3E1A3FF5BFEDBD8BBC8E40C0BF13C0F63FAABF22C0E3BF88BF8EC0BDBFB2C02EC05A3F01C0A0C0B33F07C09A3D65BF414052BF39C0DCBE783F87406440"> : tensor<1x50x3xbf16>
    %1 = stablehlo.constant dense<[[-1.929690e+00, -2.281250e+00, -2.296880e+00]]> : tensor<1x3xbf16>
    return %0, %1 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> tensor<1x50x3xbf16> {
    %0 = stablehlo.constant dense<"0x96BED5C00DC01A40BEC03C409BBFD1BFBA3EE8400C408EBE1CC005C01BBECABF253F9440E3BD813F23C08940B83F8440244091BC03C0A8BEDE3D51BF94C083C08CBFF74024C00B40CEBE343E73BFB8BF8B40C8BF2CC0914073C0A6C0CE3F39BF80406AC06ABFF0C082C01BC0FDC083C0E73E653FB93F9E3FB340593FA03E373FABC03140B23FC74041402AC082C0CC3F88C09A402E4160BF1C4018C1C0BF93403540EC3D5040E63E08401BBF8F4090C0733F503F953E76C0733FCD3F1BC013BF58C0B4BF88C09FBD07C01FC015402340134062BFB8BFACC06B3FE53F9D3E73C00FBE40407CC0B74093BF03BFFCBFAF3E1A3FF5BFEDBD8BBC8E40C0BF13C0F63FAABF22C0E3BF88BF8EC0BDBFB2C02EC05A3F01C0A0C0B33F07C09A3D65BF414052BF39C0DCBE783F87406440"> : tensor<1x50x3xbf16>
    return %0 : tensor<1x50x3xbf16>
  }
}

