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
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xbf16>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1x125xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xbf16>, tensor<1xbf16>) {
    %0 = stablehlo.constant dense<"0xD13CD7BF12C01240AD3E6AC0EBBE05C0A93FFA3F5DBF6C40A63E09C061404BBF6DBFAD3FA7C0AE40D03E0F3FD3BF41C0824039BF13C057C006BB384068C07040133F2C40B93F1940D83FC4C09C3FFF3DCB3F8240BA40AEC0CA3ECD3E8D40E03F8D40E13FD0BF9F409C4071402EBFD23E6040B73F9C401040ABBFD1BFF7C08D3DB840523FDFBE4C40D8BD98BF50C09EBF643F4EC098400BBF4940A340623EE33D38BF484084BFC63F7940A53ECABEA4BF1541A73F7CC0ADBEC7BF3940ABBF544001C1CCBF06406A40893FFC3E983F0DC0D5BE0B40D93F2EC033BC8DBEC63F4CC0B1BF89BEA140FF3F02409CC0F53FA8BFEC4012C0794091C0A0C0"> : tensor<1x125xbf16>
    %1 = stablehlo.constant dense<-3.343750e+00> : tensor<1xbf16>
    return %0, %1 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> tensor<1x125xbf16> {
    %0 = stablehlo.constant dense<"0x54C0D7BF12C01240AD3E6AC0EBBE05C0A93FFA3F5DBF6C40A63E09C061404BBF6DBFAD3FA7C0AE40D03E0F3FD3BF41C0824039BF13C057C006BB384068C07040133F2C40B93F1940D83FC4C09C3FFF3DCB3F8240BA40AEC0CA3ECD3E8D40E03F8D40E13FD0BF9F409C4071402EBFD23E6040B73F9C401040ABBFD1BFF7C08D3DB840523FDFBE4C40D8BD98BF50C09EBF643F4EC098400BBF4940A340623EE33D38BF484084BFC63F7940A53ECABEA4BF1541A73F7CC0ADBEC7BF3940ABBF544001C1CCBF06406A40893FFC3E983F0DC0D5BE0B40D93F2EC033BC8DBEC63F4CC0B1BF89BEA140FF3F02409CC0F53FA8BFEC4012C0794091C0A0C0"> : tensor<1x125xbf16>
    return %0 : tensor<1x125xbf16>
  }
}

