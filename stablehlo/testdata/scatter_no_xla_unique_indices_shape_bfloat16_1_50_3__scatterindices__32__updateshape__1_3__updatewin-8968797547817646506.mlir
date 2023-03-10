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
    %0 = stablehlo.constant dense<"0x0DC03B3F2AC0124086406E3E7DC079BE413F44402EBF993FC5BF26BFCD3F4DBEB0BF51BFA4C05640E43E1B40ADBF88BFD93F813FAFBF8340D2BE52C0E93EE240C3BF4D3D8C408A3EA6C09FC0AB3EFD3F19401C40CD3F22BFF1BFBC3F53BFBD408FC093BF843F6CC0FBBEB4BF3DBEDEBF144016C036403FC02BC030C02E400CC06640B740453F03C076C0B63F75C098C016C0EDBF143F56C088C0563F8B40F0BDEDBC6940173E88BF363F8340BA4052BF49BFB93EA3BF5E40A7BF2BC01EBC0440EDBFB3BF8CC00E3F2B40893FEFBFBA3F4BC0883D19C0CFBE884076BF04C0E93CDB3F11405E4091C004C0A5BF534080401DC0C5BED9BF0240193FB83F83C070C0C53FE0BE2340EABF27C197BE87405DC038C0934027C0424020C025C0E13FCE3F5A4096BF9D3FC6BED9BF453F"> : tensor<1x50x3xbf16>
    %1 = stablehlo.constant dense<[[-3.984380e-01, -2.080080e-01, -3.421880e+00]]> : tensor<1x3xbf16>
    return %0, %1 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> tensor<1x50x3xbf16> {
    %0 = stablehlo.constant dense<"0x0DC03B3F2AC0124086406E3E7DC079BE413F44402EBF993FC5BF26BFCD3F4DBEB0BF51BFA4C05640E43E1B40ADBF88BFD93F813FAFBF8340D2BE52C0E93EE240C3BF4D3D8C408A3EA6C09FC0AB3EFD3F19401C40CD3F22BFF1BFBC3F53BFBD408FC093BF843F6CC0FBBEB4BF3DBEDEBF144016C036403FC02BC030C02E400CC06640B740453F03C076C0B63F75C098C016C0EDBF143F56C088C0563F8B40F0BDEDBC6940173E88BF363F8340BA4052BF49BFB93EA3BF5E40A7BF2BC01EBC0440CCBE55BE5BC00E3F2B40893FEFBFBA3F4BC0883D19C0CFBE884076BF04C0E93CDB3F11405E4091C004C0A5BF534080401DC0C5BED9BF0240193FB83F83C070C0C53FE0BE2340EABF27C197BE87405DC038C0934027C0424020C025C0E13FCE3F5A4096BF9D3FC6BED9BF453F"> : tensor<1x50x3xbf16>
    return %0 : tensor<1x50x3xbf16>
  }
}

