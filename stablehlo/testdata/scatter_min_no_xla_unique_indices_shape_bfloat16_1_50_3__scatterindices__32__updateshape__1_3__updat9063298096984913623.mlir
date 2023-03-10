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
    %0 = stablehlo.constant dense<"0xBEBF5B3E90BEFBBFE1BEC240153F2240D4BF0840513C16C0E34090BEA6BF67BEC5C018BF974057C0E1BF80404140C53F7540D6BF2E40F6BF8BC0634003406F405A408040BFBF95C0B13F464061C0B4BF1A40CD3F0BC0613FD7C003C086402E40DFBECCBF20C095BF7ABF393F68BFC43F7CC01CBE054076BA4CBFE0C04CBF3A3EA9C00EC0F63F47C037BF9F3FEF3DE83F74C088408E3FC8C085BD2040D240A640393FA4BFE8C0A53F0F417D40BCBFC7C075C03C40C6C055C0CEBF47C0DB3FE8BDA93E18BF7B401F3E4440EB3F1B4007C02F4097C0F1BFF03E42408A3E3DC00DC0453E8DBF843C3CC007402BC034C0A13F5F3E4B40A23E2CC074BF5DC00E40194003C0F8BD96BF96BF66C01BC091C0903F1940204094C043C098BFC4C097BF5AC0DDBE5540FD3E75400240ABBD"> : tensor<1x50x3xbf16>
    %1 = stablehlo.constant dense<[[1.906250e+00, 1.031250e+00, -9.218750e-01]]> : tensor<1x3xbf16>
    return %0, %1 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> tensor<1x50x3xbf16> {
    %0 = stablehlo.constant dense<"0xBEBF5B3E90BEFBBFE1BEC240153F2240D4BF0840513C16C0E34090BEA6BF67BEC5C018BF974057C0E1BF80404140C53F7540D6BF2E40F6BF8BC0634003406F405A408040BFBF95C0B13F464061C0B4BF1A40CD3F0BC0613FD7C003C086402E40DFBECCBF20C095BF7ABF393F68BFC43F7CC01CBE054076BA4CBFE0C04CBF3A3EA9C00EC0F63F47C037BF9F3FEF3DE83F74C088408E3FC8C085BD2040D240A640393FA4BFE8C0A53F0F417D40BCBFC7C075C03C40C6C055C0CEBF47C0DB3FE8BDA93E18BF6CBF1F3E4440EB3F1B4007C02F4097C0F1BFF03E42408A3E3DC00DC0453E8DBF843C3CC007402BC034C0A13F5F3E4B40A23E2CC074BF5DC00E40194003C0F8BD96BF96BF66C01BC091C0903F1940204094C043C098BFC4C097BF5AC0DDBE5540FD3E75400240ABBD"> : tensor<1x50x3xbf16>
    return %0 : tensor<1x50x3xbf16>
  }
}

