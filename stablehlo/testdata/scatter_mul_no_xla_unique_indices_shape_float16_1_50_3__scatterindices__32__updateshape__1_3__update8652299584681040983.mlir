// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xf16>, tensor<1x3xf16>)
    %2 = call @expected() : () -> tensor<1x50x3xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xf16>, tensor<1xi32>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16>, tensor<1x3xf16>) {
    %0 = stablehlo.constant dense<"0x15AD27C157BC64C33E2DACC6EE4114B1DCC5CC429BBB3FBA5A3F703DEC441540EF3F3746AF40B3395A3AB4BF63412F3DF2BF2E4402BAF7410F409D4059C5ADC13E31824708C136B42A39DEC4BDC129B95A3837C1C5B871AFB5B4A1419746763DDCBD96BB6BC313C244321E3B93B9973C4E4488C0DBBC943C0E43AE32093769B820BB3DBD86BB2B46A248F4BD3D36BBBD37C4513CE743FC454F3368BF8343D137DA44CC412244434219B616401D3E1F38F337033D2FC465B40844E6BF8C3DBDBF88BEB642EA32CCBF98C0F92D71C00E410BC766B89FBED3C59C45A9BA224359C2163B483258BD5C338BC0EA3E45BCFDB1623B1A3DAC45CA39E1398AC5B93FCA3C36B91645B9429CB970BD13C19FC36ABE41BCC7BCD14546C2B1C0BAC010C1D03D2F3DEBC4B3BBED3975BB40C4"> : tensor<1x50x3xf16>
    %1 = stablehlo.constant dense<[[4.546880e+00, 4.761720e+00, 1.632810e+00]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x50x3xf16> {
    %0 = stablehlo.constant dense<"0x15AD27C157BC64C33E2DACC6EE4114B1DCC5CC429BBB3FBA5A3F703DEC441540EF3F3746AF40B3395A3AB4BF63412F3DF2BF2E4402BAF7410F409D4059C5ADC13E31824708C136B42A39DEC4BDC129B95A3837C1C5B871AFB5B4A1419746763DDCBD96BB6BC313C244321E3B93B9973C4E4488C0DBBC943C0E43AE32093769B820BB3DBD86BB2B46A248F4BD3D36BBBD37C4513CE743FC454F3368BF8343D137DA44CC412244434219B616401D3E1F38F337033D2FC465B40844E6BF8C3DBDBF6DC7FD4BA535CCBF98C0F92D71C00E410BC766B89FBED3C59C45A9BA224359C2163B483258BD5C338BC0EA3E45BCFDB1623B1A3DAC45CA39E1398AC5B93FCA3C36B91645B9429CB970BD13C19FC36ABE41BCC7BCD14546C2B1C0BAC010C1D03D2F3DEBC4B3BBED3975BB40C4"> : tensor<1x50x3xf16>
    return %0 : tensor<1x50x3xf16>
  }
}

