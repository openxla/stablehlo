// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>)
    %2 = call @expected() : () -> tensor<4x2x3x5xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xbf16>, tensor<2xi32>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) {
    %0 = stablehlo.constant dense<"0xB63F13C08CBFEA3DCBBFC24046C05240AC3FBE3F6640BCBF07BF06C0ADC0864019C053C068C0D840CA3CFB3F303F84C0AC40963FAF3F8B3FA73FD93C2F3EA63F92BE4BC00A3F9040E7401140E4BF30402F408B3F2ABF803F123C2E3E384008403EC03CC0773FFA3F99C0334003BE86C0343E7240C34057402E3F913FBD3E12401BC092407F40B3BF513F2C40FCC0CF3F61C048BF71C0113F58BF0140A9BF84C014C008C1E63F09C117C0364084C0B5BFB4C0063F2640BEBFC440A0BF1F3F45C02040273F63BFF840E53F86BF18C06EBE003EA23E9ABE2140A0C080405F3F38BFAABF28C082403C407F3E4FC06CBF12C0"> : tensor<4x2x3x5xbf16>
    %1 = stablehlo.constant dense<[[4.687500e-01, -3.984380e-01, -7.343750e-01], [4.093750e+00, -1.429690e+00, 3.390630e+00], [-8.117670e-03, 1.296880e+00, 3.281250e+00], [-2.515630e+00, 6.406250e-01, 3.765630e+00]]> : tensor<4x3xbf16>
    return %0, %1 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xbf16> {
    %0 = stablehlo.constant dense<"0xB63F13C08CBFEA3DCBBFC24046C05240AC3FCCBE6640BCBF07BF06C0ADC0864019C053C068C0D840CA3CFB3F303F84C0AC40963FAF3F8B3FA73FD93C2F3EA63F92BE4BC00A3F9040E7401140E4BFB7BF2F408B3F2ABF803F123C2E3E384008403EC03CC0773FFA3F99C0334003BE86C0343E7240C34057402E3F913FBD3E12401BC092407F40B3BF513FA63FFCC0CF3F61C048BF71C0113F58BF0140A9BF84C014C008C1E63F09C117C0364084C0B5BFB4C0063F2640BEBFC440A0BF21C045C02040273F63BF243FE53F86BF18C06EBE003EA23E9ABE2140A0C080405F3F38BFAABF28C082403C407F3E4FC06CBF12C0"> : tensor<4x2x3x5xbf16>
    return %0 : tensor<4x2x3x5xbf16>
  }
}

