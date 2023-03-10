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
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xbf16>, tensor<1xi32>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>) {
    %0 = stablehlo.constant dense<"0x8B401FC06840EFBFA6BE44C09ABFBBC07D40B73F62C085405EC0E8BFA240E53E434004C0463FA4C0C5408E402EC07BC0CEC0CD3E99C005BE33C02E3F1941D03FD3BF414052C05CBF62400ABF5E4030404240E73F18BFD8BEEBBF3F4019BF643F09C051C076C04440E7C06EC02D40DE3F51BFEC3F37C194BF3ABF76402940B5BFBD3FFE3F63C0A3C025C02740F5BFFEC05640654053C068409C40353FA040123F0AC0AF4046C02F4041C0193F903F6D4041C0663F524071403A40ACBF254055BF383FB8402640AE3F03C1593F2DBF9D409CBFA3C09C40A23E94BF2B402A3F6EC0E6BFA44071409ABFCF3E444000404240A3C0BF3FF040953F34C05D3F6EBF42405BC09D3F4CBFF6BFB3BF9BC0023DBBBF033FA03F0FC0DABFD43FB53F844006C0274003C009C19C40803FB640"> : tensor<1x50x3xbf16>
    %1 = stablehlo.constant dense<[[-7.851560e-01, 3.421880e+00, -2.718750e+00]]> : tensor<1x3xbf16>
    return %0, %1 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> tensor<1x50x3xbf16> {
    %0 = stablehlo.constant dense<"0x8B401FC06840EFBFA6BE44C09ABFBBC07D40B73F62C085405EC0E8BFA240E53E434004C0463FA4C0C5408E402EC07BC0CEC0CD3E99C005BE33C02E3F1941D03FD3BF414052C05CBF62400ABF5E4030404240E73F18BFD8BEEBBF3F4019BF643F09C051C076C04440E7C06EC02D40DE3F51BFEC3F37C194BF3ABF76402940B5BFBD3FFE3F63C0A3C025C02740F5BFFEC05640654053C068409C40353FA040123F0AC0AF4046C02F4041C0193F903F6D4041C0663F524071403A40ACBF254055BF88BD134100BEAE3F03C1593F2DBF9D409CBFA3C09C40A23E94BF2B402A3F6EC0E6BFA44071409ABFCF3E444000404240A3C0BF3FF040953F34C05D3F6EBF42405BC09D3F4CBFF6BFB3BF9BC0023DBBBF033FA03F0FC0DABFD43FB53F844006C0274003C009C19C40803FB640"> : tensor<1x50x3xbf16>
    return %0 : tensor<1x50x3xbf16>
  }
}

