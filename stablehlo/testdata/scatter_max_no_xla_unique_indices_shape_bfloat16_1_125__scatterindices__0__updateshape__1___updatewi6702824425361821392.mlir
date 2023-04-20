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
    %0 = stablehlo.constant dense<"0x943F3A400F3DD5C088BEB0BFF03F8EBF3040743E30BFB23F2FC0B63F453FB6C0DA3E743F5DBF34C0FEBF714036408ABE7C3FAC40A2BFCA40403F9EC03AC0A740B3C0B9BF67BF09C0E9BF77BF1F3D99404F409FBFAB40674019408F40C1BFFE3FE8C0A0BF79407B3E42BF2A4061403540BABE0CC066BF22C009BF3C408EBD6F40D73F63401CC0B34047C0593E0CC0B5BD72BF14C0413F24C040C01240DCBEF03F85BF94C0953F6AC0853FB1BF0140BF3F9840D2C05140B0BF184051C0C1BE5EBF0F409A40074198BF41C0B1BE8BC05CC0E4BD424090C0D6C06CC085C0EBC0323F3CC04CC0CE3E574026C01FC1574017BE46408440EBBDC9BF2B40"> : tensor<1x125xbf16>
    %1 = stablehlo.constant dense<-4.593750e+00> : tensor<1xbf16>
    return %0, %1 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> tensor<1x125xbf16> {
    %0 = stablehlo.constant dense<"0x943F3A400F3DD5C088BEB0BFF03F8EBF3040743E30BFB23F2FC0B63F453FB6C0DA3E743F5DBF34C0FEBF714036408ABE7C3FAC40A2BFCA40403F9EC03AC0A740B3C0B9BF67BF09C0E9BF77BF1F3D99404F409FBFAB40674019408F40C1BFFE3FE8C0A0BF79407B3E42BF2A4061403540BABE0CC066BF22C009BF3C408EBD6F40D73F63401CC0B34047C0593E0CC0B5BD72BF14C0413F24C040C01240DCBEF03F85BF94C0953F6AC0853FB1BF0140BF3F9840D2C05140B0BF184051C0C1BE5EBF0F409A40074198BF41C0B1BE8BC05CC0E4BD424090C0D6C06CC085C0EBC0323F3CC04CC0CE3E574026C01FC1574017BE46408440EBBDC9BF2B40"> : tensor<1x125xbf16>
    return %0 : tensor<1x125xbf16>
  }
}

