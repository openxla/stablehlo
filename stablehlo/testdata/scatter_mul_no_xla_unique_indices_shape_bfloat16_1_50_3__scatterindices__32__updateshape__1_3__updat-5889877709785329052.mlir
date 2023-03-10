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
    %0 = stablehlo.constant dense<"0x934034C09DBF17C00B4083BF2040853EA93F473F2B40983FC6BF21402AC0BCBF30C063402B3EDC3F1240F1BE8CC0DE3E49BF51C0D4400840EBBF2C4011C02ABE98C08CBF454069BF6FC01EC0A9BF5F3F0A40AC3FA1BFBABFA140E83FB13F83BFA93E0C3F33401ABF3E3FA2C054402D40273FC2BF91BEA13F24C0633EE73FBA408FBF0EC005C07AC023C0D93F8EBE0C401BC0E63F5140663ED13EDCBF7F406E40D5BF10406C40C03FBB4093C02BBF584009C0B7BF824083BEADBF3AC098C00FC06140524008413DC05540AABFBDC06AC090400EC0883E273F83BF35C037C076C0D03E9340F9BFF13F0A3F48408BBF7E3F053F89C08CC07D40163F8E40CBBF73400E407D3F88BF673DE43FEEBF06C047BF3DC0383F0A3E8540D03EBDBED03E55C0DCC0CDC023C0DF3F52C06840"> : tensor<1x50x3xbf16>
    %1 = stablehlo.constant dense<[[2.046880e+00, 1.320310e+00, -3.812500e+00]]> : tensor<1x3xbf16>
    return %0, %1 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> tensor<1x50x3xbf16> {
    %0 = stablehlo.constant dense<"0x934034C09DBF17C00B4083BF2040853EA93F473F2B40983FC6BF21402AC0BCBF30C063402B3EDC3F1240F1BE8CC0DE3E49BF51C0D4400840EBBF2C4011C02ABE98C08CBF454069BF6FC01EC0A9BF5F3F0A40AC3FA1BFBABFA140E83FB13F83BFA93E0C3F33401ABF3E3FA2C054402D40273FC2BF91BEA13F24C0633EE73FBA408FBF0EC005C07AC023C0D93F8EBE0C401BC0E63F5140663ED13EDCBF7F406E40D5BF10406C40C03FBB4093C02BBF584009C0B7BF824083BEADBF3AC098C00FC0E6408B4002C23DC05540AABFBDC06AC090400EC0883E273F83BF35C037C076C0D03E9340F9BFF13F0A3F48408BBF7E3F053F89C08CC07D40163F8E40CBBF73400E407D3F88BF673DE43FEEBF06C047BF3DC0383F0A3E8540D03EBDBED03E55C0DCC0CDC023C0DF3F52C06840"> : tensor<1x50x3xbf16>
    return %0 : tensor<1x50x3xbf16>
  }
}

