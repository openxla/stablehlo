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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xbf16>, tensor<1xi32>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>) {
    %0 = stablehlo.constant dense<"0x3DC0B2BFA3BF0ABEF4BD944080BF19BFFCBF13C0BABE183F45401B3E544077C0F33FDEBE843ED03F0440263F8AC03740F3BF24C0724030BF603FA7BF00BC074035C027C085BD4BBF1C3FDF3F9040CFC013401EBF2FC0703F77C0A340523E77C05840B1BE3B3F1A4070BF45BF314089BFB1BFC0BF754072C073C08B404540E9BFFA3F473F72C00940AD40F4BF79BE63BE31C03740793EA93FB43FCDC02B40BEC09EBF834071BFE33F24400CBEBD3F2FC0543F624087C0DABF5540AB40363EC5BF90BF443FD73E6540B4BD64C05140FF3FDC40093D27BD273E86C0B53F08C03D4020C069BF9B40EEC052C08D409ABFB6C05F402340753FE6C0D6BF82BF09C0F23E34C08D3F9C3F76BE5CC08E4097BF49BF40C0913F86C00A4161BF2CBF674019408B401B3FB1BE8CC0A14081BF"> : tensor<1x50x3xbf16>
    %1 = stablehlo.constant dense<[[-3.328130e+00, 3.296880e+00, 1.265630e+00]]> : tensor<1x3xbf16>
    return %0, %1 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> tensor<1x50x3xbf16> {
    %0 = stablehlo.constant dense<"0x3DC0B2BFA3BF0ABEF4BD944080BF19BFFCBF13C0BABE183F45401B3E544077C0F33FDEBE843ED03F0440263F8AC03740F3BF24C0724030BF603FA7BF00BC074035C027C085BD4BBF1C3FDF3F9040CFC013401EBF2FC0703F77C0A340523E77C05840B1BE3B3F1A4070BF45BF314089BFB1BFC0BF754072C073C08B404540E9BFFA3F473F72C00940AD40F4BF79BE63BE31C03740793EA93FB43FCDC02B40BEC09EBF834071BFE33F24400CBEBD3F2FC0543F624087C0DABF5540AB40363EC5BF70402240083F6540B4BD64C05140FF3FDC40093D27BD273E86C0B53F08C03D4020C069BF9B40EEC052C08D409ABFB6C05F402340753FE6C0D6BF82BF09C0F23E34C08D3F9C3F76BE5CC08E4097BF49BF40C0913F86C00A4161BF2CBF674019408B401B3FB1BE8CC0A14081BF"> : tensor<1x50x3xbf16>
    return %0 : tensor<1x50x3xbf16>
  }
}

