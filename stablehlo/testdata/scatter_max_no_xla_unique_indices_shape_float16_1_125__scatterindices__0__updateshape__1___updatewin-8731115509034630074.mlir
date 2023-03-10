// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xf16>, tensor<1xf16>)
    %2 = call @expected() : () -> tensor<1x125xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xf16>, tensor<1xi32>, tensor<1xf16>) -> tensor<1x125xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xf16>, tensor<1x125xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xf16>, tensor<1xf16>) {
    %0 = stablehlo.constant dense<"0x6ABE8C41A73D4D3EBE3EBD3CCE43BBBEFD3892B97EB9F940AEC28AC191B680BB34C3CCB9B83A703DCB3447BB3F2241403F44D6B76A3DACB82741623FAD3C97BF6D39973D4840583E9D453645DD3B713CC1402C467FBE69413A43D12B35AE4C3F83402E3A2CC095C22ABFB035A835C3B44C3C4240AA453A3B0AC3EF2E5CC0B7BB822E8A45DDBBA03C32C4F13ACFC0C83F8BC24EB81FB8BBBB1F4273419A43F23FE2C09B3F95410D4273BFC5BF06BF4D4653C8B63BE9C4434381C6993D7DBC2B437D381E3D563DB53AA241D9449A3FED4119393FC1A8409B3FBA3B8C403E44033D4FC22C41BD3AFA40BC44A13CCD3A63BB654053421DB4EDB82EBB"> : tensor<1x125xf16>
    %1 = stablehlo.constant dense<-1.838870e+00> : tensor<1xf16>
    return %0, %1 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> tensor<1x125xf16> {
    %0 = stablehlo.constant dense<"0x6ABE8C41A73D4D3EBE3EBD3CCE43BBBEFD3892B97EB9F940AEC28AC191B680BB34C3CCB9B83A703DCB3447BB3F2241403F44D6B76A3DACB82741623FAD3C97BF6D39973D4840583E9D453645DD3B713CC1402C467FBE69413A43D12B35AE4C3F83402E3A2CC095C22ABFB035A835C3B44C3C4240AA453A3B0AC3EF2E5CC0B7BB822E8A45DDBBA03C32C4F13ACFC0C83F8BC24EB81FB8BBBB1F4273419A43F23FE2C09B3F95410D4273BFC5BF06BF4D4653C8B63BE9C4434381C6993D7DBC2B437D381E3D563DB53AA241D9449A3FED4119393FC1A8409B3FBA3B8C403E44033D4FC22C41BD3AFA40BC44A13CCD3A63BB654053421DB4EDB82EBB"> : tensor<1x125xf16>
    return %0 : tensor<1x125xf16>
  }
}

