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
    %0 = stablehlo.constant dense<"0xA0BF9BC067BE44C06D3F463F81C0D7BFAB405AC0BDBF3F40023CCFBFBB3FA73F3ABF053FBEBFDF3F86C01140C1C0B6C0C1BFB0BEDBBE07C005C03CC0873F1DC086BF3640C23F0940163D174020C019BF6BBC2EBFF4BFEBC02F40264064C0BFBDF8C004C0C4C0C9BF8040983E2040BABF70BED5BC104099BF47BF0AC08CBFB63FA2C075BFBFC08A3D8DC0CABF92406FC03940A0C0E53F01C048C009C0B2BF7740B0BFFA3E23C039BF37C0BB3F18C08BC01DBFA4C06FBF4DC02740D940504018C0AB3E4FBF9FC040BF46C0D0C0EC3FB8400B3EA9BF9440EFBFDA3F8E3EB23F9D405CC03440FC3FCF3F85BF9E400DBF1F4003407D40CDC07440D8BE"> : tensor<1x125xbf16>
    %1 = stablehlo.constant dense<-4.468750e+00> : tensor<1xbf16>
    return %0, %1 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> tensor<1x125xbf16> {
    %0 = stablehlo.constant dense<"0x8FC09BC067BE44C06D3F463F81C0D7BFAB405AC0BDBF3F40023CCFBFBB3FA73F3ABF053FBEBFDF3F86C01140C1C0B6C0C1BFB0BEDBBE07C005C03CC0873F1DC086BF3640C23F0940163D174020C019BF6BBC2EBFF4BFEBC02F40264064C0BFBDF8C004C0C4C0C9BF8040983E2040BABF70BED5BC104099BF47BF0AC08CBFB63FA2C075BFBFC08A3D8DC0CABF92406FC03940A0C0E53F01C048C009C0B2BF7740B0BFFA3E23C039BF37C0BB3F18C08BC01DBFA4C06FBF4DC02740D940504018C0AB3E4FBF9FC040BF46C0D0C0EC3FB8400B3EA9BF9440EFBFDA3F8E3EB23F9D405CC03440FC3FCF3F85BF9E400DBF1F4003407D40CDC07440D8BE"> : tensor<1x125xbf16>
    return %0 : tensor<1x125xbf16>
  }
}

