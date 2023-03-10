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
      stablehlo.return %arg1 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xbf16>, tensor<1xi32>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>) {
    %0 = stablehlo.constant dense<"0xE3C06D3F404014BF10C060C01040E4BFE4BF5B3FD6BF823F24BC863FA3BE5ABF04BF74408140A6BF1CBF923F21408F4071C0F7C03F40DEC084C075C0FCBE9F40743FC1BF06C04B401FBFD8C01640DEBEF8BF533F12C0A5C0933F9D40BA3F973FBCBD52C0D6C08E40BEBF873EC4BE3540D1BF93404AC0FF3EEE3F214099BFBE40ABBF27C02A3F1D4055BF7AC0D3C0E53D66C052C03B4036C00FBE3940823F293D86BF2540C23F313E01C009C04C407E405C3ED2C0EBBF52404F401A4071BF9A3E46BFF33ECDC01BC0A5BFBF3EA1C0ABC057C0D8BF914014BFF03F6EC0A13E55C005C08DBF1A4013BE80BF4FC061C06B4039C0EE3F01BFE4BFDD3F8C3E8FC0A33F09C0B2BF5B4009C080C01D3F7E3F80BF553E29BF123E8E3F453F5BC05740A8BFEF4090C09540094022403140"> : tensor<1x50x3xbf16>
    %1 = stablehlo.constant dense<[[7.718750e+00, 1.914060e+00, -2.281250e+00]]> : tensor<1x3xbf16>
    return %0, %1 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> tensor<1x50x3xbf16> {
    %0 = stablehlo.constant dense<"0xE3C06D3F404014BF10C060C01040E4BFE4BF5B3FD6BF823F24BC863FA3BE5ABF04BF74408140A6BF1CBF923F21408F4071C0F7C03F40DEC084C075C0FCBE9F40743FC1BF06C04B401FBFD8C01640DEBEF8BF533F12C0A5C0933F9D40BA3F973FBCBD52C0D6C08E40BEBF873EC4BE3540D1BF93404AC0FF3EEE3F214099BFBE40ABBF27C02A3F1D4055BF7AC0D3C0E53D66C052C03B4036C00FBE3940823F293D86BF2540C23F313E01C009C04C407E405C3ED2C0EBBF52404F401A4071BF9A3EF740F53F12C01BC0A5BFBF3EA1C0ABC057C0D8BF914014BFF03F6EC0A13E55C005C08DBF1A4013BE80BF4FC061C06B4039C0EE3F01BFE4BFDD3F8C3E8FC0A33F09C0B2BF5B4009C080C01D3F7E3F80BF553E29BF123E8E3F453F5BC05740A8BFEF4090C09540094022403140"> : tensor<1x50x3xbf16>
    return %0 : tensor<1x50x3xbf16>
  }
}

