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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xbf16>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1x125xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xbf16>, tensor<1xbf16>) {
    %0 = stablehlo.constant dense<"0x863F67C0BAC0D1C055BE8C40CB403D4038BF26C0D3BFC8BF50C099C080BFE53E5D407B4015C064BF5CC0163FDCC090BF9F3F933FACBE993F173F0FC06C4082407DBF8F408B40294053C0143F22C00C4011408DBE6FBD483F4940E9BFD1BF8CBFAC3F93BF313EAEC00C418EBF993E353E9BC0894093C0B2BEE23F2A3F3940A0404ABF193DEFC0CCC05240843FFABF3EBF94BFDBBF6A3ED73DEC3E0A3F68C01DC0C8BF06C0BCC0B9BE05C03F40894054BE96BF6BC043C0FC3F47406040EC3DA44010C0183F6BC0AEC02FC0AB3F8CBFB8BF24BF523F04407FC0D83FEE3F37C0CA3F16BF0240FBBCE0BE75BF73BFC0BF3DC0E93F5340613F2640243F"> : tensor<1x125xbf16>
    %1 = stablehlo.constant dense<-5.390630e-01> : tensor<1xbf16>
    return %0, %1 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> tensor<1x125xbf16> {
    %0 = stablehlo.constant dense<"0x863F67C0BAC0D1C055BE8C40CB403D4038BF26C0D3BFC8BF50C099C080BFE53E5D407B4015C064BF5CC0163FDCC090BF9F3F933FACBE993F173F0FC06C4082407DBF8F408B40294053C0143F22C00C4011408DBE6FBD483F4940E9BFD1BF8CBFAC3F93BF313EAEC00C418EBF993E353E9BC0894093C0B2BEE23F2A3F3940A0404ABF193DEFC0CCC05240843FFABF3EBF94BFDBBF6A3ED73DEC3E0A3F68C01DC0C8BF06C0BCC0B9BE05C03F40894054BE96BF6BC043C0FC3F47406040EC3DA44010C0183F6BC0AEC02FC0AB3F8CBFB8BF24BF523F04407FC0D83FEE3F37C0CA3F16BF0240FBBCE0BE75BF73BFC0BF3DC0E93F5340613F2640243F"> : tensor<1x125xbf16>
    return %0 : tensor<1x125xbf16>
  }
}

