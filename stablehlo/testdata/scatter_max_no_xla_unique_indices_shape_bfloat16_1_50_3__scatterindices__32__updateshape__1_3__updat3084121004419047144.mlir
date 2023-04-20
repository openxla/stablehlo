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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xbf16>, tensor<1xi32>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>) {
    %0 = stablehlo.constant dense<"0x11BF1E407DBF84C0284003BF84BFDCBFFC3FC9BE35409FBF3BC0D7BE88BE9640A8BF8BC069C094BE84BF87BE474063C02DC0C240F83F2840C73FC7BE94C05BC059C099C0CA3FD640853ED83FD6BF744094C0ACBC303F1D40823ECABF143FB4C00C3EABBFEE3F2540A63F2240AE40A7BEBEBE02BE27C092BF8C4087408E3F2D401AC084C043C0B9402AC0373F5BC035C02F3F513FE6BFDE3F7FC069C063C001C0C8BF01C07340AFBFB2BFC43FDEBF113FA8C04540FC3F4BC03C4033407AC09EC060C0B240C33FADBF42402B3F60BF6F3E483F084062BFFB3F4C4092406AC09FBFDA3F2BC01A4034BE8C40B23FFFBF3A3F0B3FBDBFB6C041BFDCC03E407440FBBFDCBEA6BF2D4060C0733FB23F283F8CBE54C029C04DC038408D409CC013C093BE2A4040C04040B1BFF94020C0"> : tensor<1x50x3xbf16>
    %1 = stablehlo.constant dense<[[-2.937500e+00, 2.218750e+00, 1.203130e+00]]> : tensor<1x3xbf16>
    return %0, %1 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> tensor<1x50x3xbf16> {
    %0 = stablehlo.constant dense<"0x11BF1E407DBF84C0284003BF84BFDCBFFC3FC9BE35409FBF3BC0D7BE88BE9640A8BF8BC069C094BE84BF87BE474063C02DC0C240F83F2840C73FC7BE94C05BC059C099C0CA3FD640853ED83FD6BF744094C0ACBC303F1D40823ECABF143FB4C00C3EABBFEE3F2540A63F2240AE40A7BEBEBE02BE27C092BF8C4087408E3F2D401AC084C043C0B9402AC0373F5BC035C02F3F513FE6BFDE3F7FC069C063C001C0C8BF01C07340AFBFB2BFC43FDEBF113FA8C04540FC3F4BC03C4033407AC09EC03CC0B240C33FADBF42402B3F60BF6F3E483F084062BFFB3F4C4092406AC09FBFDA3F2BC01A4034BE8C40B23FFFBF3A3F0B3FBDBFB6C041BFDCC03E407440FBBFDCBEA6BF2D4060C0733FB23F283F8CBE54C029C04DC038408D409CC013C093BE2A4040C04040B1BFF94020C0"> : tensor<1x50x3xbf16>
    return %0 : tensor<1x50x3xbf16>
  }
}

