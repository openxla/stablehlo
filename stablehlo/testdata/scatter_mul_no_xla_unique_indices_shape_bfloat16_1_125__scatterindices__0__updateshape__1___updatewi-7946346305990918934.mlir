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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xbf16>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1x125xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xbf16>, tensor<1xbf16>) {
    %0 = stablehlo.constant dense<"0x8F4020C025409C3F75BFD3BF8340AEC02D40883FD4BE8E40274047C084BF50C00BBFE5BFE23FAC40BDBD45BF12BF22C0A640AABF2140433F4AC096C0BA3EB6BF843F8F3FA33ED03F9E3F2A3F0240F73F70C08D40B3BFAA400340E5C001C0733D713FA2BF49BF88C060408640E73F4640C6C00AC0B0BB69BFD43FB13F8040843EEE3FF13F8B3F883F973F893F73C0AABF03BE7B3F1D400B40DDBF5E40B8BD11BF1C4002BFADBF2AC03040904038403FC00A401DC0B03F6C3FAFBF913F9A40973FBD3CD23D123E21BEC03E784005C0D03F5D3F99C0DE3EF8BE0BC0A03F9AC0DB3F3BC048C00E40B940C0C096BF8140253F24BF15C009BF25C089C0"> : tensor<1x125xbf16>
    %1 = stablehlo.constant dense<-2.000000e+00> : tensor<1xbf16>
    return %0, %1 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> tensor<1x125xbf16> {
    %0 = stablehlo.constant dense<"0x0FC120C025409C3F75BFD3BF8340AEC02D40883FD4BE8E40274047C084BF50C00BBFE5BFE23FAC40BDBD45BF12BF22C0A640AABF2140433F4AC096C0BA3EB6BF843F8F3FA33ED03F9E3F2A3F0240F73F70C08D40B3BFAA400340E5C001C0733D713FA2BF49BF88C060408640E73F4640C6C00AC0B0BB69BFD43FB13F8040843EEE3FF13F8B3F883F973F893F73C0AABF03BE7B3F1D400B40DDBF5E40B8BD11BF1C4002BFADBF2AC03040904038403FC00A401DC0B03F6C3FAFBF913F9A40973FBD3CD23D123E21BEC03E784005C0D03F5D3F99C0DE3EF8BE0BC0A03F9AC0DB3F3BC048C00E40B940C0C096BF8140253F24BF15C009BF25C089C0"> : tensor<1x125xbf16>
    return %0 : tensor<1x125xbf16>
  }
}

