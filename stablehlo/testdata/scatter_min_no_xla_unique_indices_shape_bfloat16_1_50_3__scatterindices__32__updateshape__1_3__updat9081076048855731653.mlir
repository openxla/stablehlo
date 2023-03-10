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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xbf16>, tensor<1xi32>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>) {
    %0 = stablehlo.constant dense<"0x284049BFA1C061BF8A3FEEC062C068400EC12E40CB4007C0443F2DBFD23FB2BE163F01C0D03F123E5EC02A404FC00FC067BF31409E4042400740E03F28C080409D4060BB9B403B3D79C08E3E8A3F883D0940E43FB1406CBF8740F0BF33BFB64017401B3F073F46C0DDBFA0BF7BC04DC017C058C0BCBEDA3FE9BF9D3F4940BCBF09BFB33FA53E87C0853FAC40ACC05340AD3F9B40E3BFD23F9AC0AF3F01BFD43CCABFD7C0DDC08D4075406F40853F9340B7BE2640983F8C40AB3FC5BF91C00E3F0BBEB2BF65C0413F833F78BF3F3D99BF814030C089BFC93F8D40533D81C00641A33E10C0B5C0BEBF853EBD3F17406240D6C0C84099BFF83F8C3F48BFA4BCD4BFC3406AC0E23FB0BF45C04440873E69BF083EEABF093F0EC07B3E954047400CBF35BFC840793F344039C0AAC0"> : tensor<1x50x3xbf16>
    %1 = stablehlo.constant dense<[[-4.238280e-01, 3.781250e+00, -2.484380e+00]]> : tensor<1x3xbf16>
    return %0, %1 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> tensor<1x50x3xbf16> {
    %0 = stablehlo.constant dense<"0x284049BFA1C061BF8A3FEEC062C068400EC12E40CB4007C0443F2DBFD23FB2BE163F01C0D03F123E5EC02A404FC00FC067BF31409E4042400740E03F28C080409D4060BB9B403B3D79C08E3E8A3F883D0940E43FB1406CBF8740F0BF33BFB64017401B3F073F46C0DDBFA0BF7BC04DC017C058C0BCBEDA3FE9BF9D3F4940BCBF09BFB33FA53E87C0853FAC40ACC05340AD3F9B40E3BFD23F9AC0AF3F01BFD43CCABFD7C0DDC08D4075406F40853F9340B7BE2640983F8C40AB3FC5BF91C00E3FD9BEB2BF65C0413F833F78BF3F3D99BF814030C089BFC93F8D40533D81C00641A33E10C0B5C0BEBF853EBD3F17406240D6C0C84099BFF83F8C3F48BFA4BCD4BFC3406AC0E23FB0BF45C04440873E69BF083EEABF093F0EC07B3E954047400CBF35BFC840793F344039C0AAC0"> : tensor<1x50x3xbf16>
    return %0 : tensor<1x50x3xbf16>
  }
}

