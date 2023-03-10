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
      stablehlo.return %arg1 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xbf16>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1x125xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xbf16>, tensor<1xbf16>) {
    %0 = stablehlo.constant dense<"0x88C0833F2EBFB8C0343FA74007BC13408ABE50BE10C00B40C04082402BC04A4019C0DDBEA1400CC019405C4070409340C33F9A3DE3BFDCBC61BFBFBFFABF3140D9BF35C01A3FBB3F5EBE1040CB3F2E40C0C0363ECC3FD3BF06BF00C08F3DA440544042C077C04BC02C4064C08FC05BBFD7BFB73EA5BC08BF85BF5E40C43E89C064BFC83D34C09D40EB3F053F194005C085BE0E404D3C933F2740AC3F4FC039BD01C00FC042BF34404DC0EF408C40E5BD224023C056C01A4098C079C07FBF673F0DC03DC049408C40B7BF43C0E03F273F5AC086C00FBE553E2E405B3FE5BEC9BF6DBE4BC021402340D13E2FC02E40BBC01A3E513F8C3F00C03640"> : tensor<1x125xbf16>
    %1 = stablehlo.constant dense<-3.062500e+00> : tensor<1xbf16>
    return %0, %1 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> tensor<1x125xbf16> {
    %0 = stablehlo.constant dense<"0x44C0833F2EBFB8C0343FA74007BC13408ABE50BE10C00B40C04082402BC04A4019C0DDBEA1400CC019405C4070409340C33F9A3DE3BFDCBC61BFBFBFFABF3140D9BF35C01A3FBB3F5EBE1040CB3F2E40C0C0363ECC3FD3BF06BF00C08F3DA440544042C077C04BC02C4064C08FC05BBFD7BFB73EA5BC08BF85BF5E40C43E89C064BFC83D34C09D40EB3F053F194005C085BE0E404D3C933F2740AC3F4FC039BD01C00FC042BF34404DC0EF408C40E5BD224023C056C01A4098C079C07FBF673F0DC03DC049408C40B7BF43C0E03F273F5AC086C00FBE553E2E405B3FE5BEC9BF6DBE4BC021402340D13E2FC02E40BBC01A3E513F8C3F00C03640"> : tensor<1x125xbf16>
    return %0 : tensor<1x125xbf16>
  }
}

