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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xbf16>, tensor<2xi32>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) {
    %0 = stablehlo.constant dense<"0x8A4056404F4086C03C4080405CC0E43FEC3F3B407D3FB1C04BC0AA3E56BF003E823E8C3FCB3E75C060C08ABF9BBF9FC0463F5FC049C090BF1440164054BFAE40E23EF7BF62BED5C097BF9BBEB9BFB840A540633FE3BF72C0FDBF94BFEA3F2740943F773EFF3F9C3F89BE82BEB83E4FC096C068400DC0813FB0C0A0403D40A03FDABE603F2A400B3F9AC059403E4015401BBFABBF744094C0093F623FA53E763F004099402B402DC0264068BF703FEB3F65C01AC012C0013FE13D66C01CC006BFEC3FA04040BFD74043C0D43E16C0B2BFB5C0383FCEBF133F874027403CC0A5BEBDC0BC3E1EC0B7C08640F53EB5BF8240"> : tensor<4x2x3x5xbf16>
    %1 = stablehlo.constant dense<[[1.585940e+00, -1.234380e+00, -3.710940e-01], [4.656250e+00, 5.375000e+00, -1.259770e-01], [-3.140630e+00, -2.625000e+00, 3.964840e-01], [-2.828130e+00, -1.367190e-01, -9.453120e-01]]> : tensor<4x3xbf16>
    return %0, %1 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xbf16> {
    %0 = stablehlo.constant dense<"0x8A4056404F4086C0954080405CC0E43FEC3F67C07D3FB1C04BC0AA3E9F3E003E823E8C3FCB3E75C060C08ABF9BBF9FC0463F5FC049C090BF1440164054BFAE40E23EF7BF84BFD5C097BF9BBEB9BFF741A540633FE3BF72C07F3E94BFEA3F2740943F773EFF3F9C3F89BE82BEB83E4FC096C068400DC0813FB0C0A0403D40A03FAB3F603F2A400B3F9AC00EC13E4015401BBFABBFC13F94C0093F623FA53E763F004099402B402DC0264068BF703FEB3F65C01AC012C0013FE13D66C0DD4006BFEC3FA04040BF6BBF43C0D43E16C0B2BFAB40383FCEBF133F874027403CC0A5BEBDC0BC3E1EC0B7C08640F53EB5BF8240"> : tensor<4x2x3x5xbf16>
    return %0 : tensor<4x2x3x5xbf16>
  }
}

