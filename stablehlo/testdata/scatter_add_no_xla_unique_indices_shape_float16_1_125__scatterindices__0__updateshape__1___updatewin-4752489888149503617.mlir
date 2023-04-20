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
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xf16>, tensor<1xi32>, tensor<1xf16>) -> tensor<1x125xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xf16>, tensor<1x125xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xf16>, tensor<1xf16>) {
    %0 = stablehlo.constant dense<"0x90C2453DF4B0F14488C2B94224AB7DC108423E42DFC524C062381A31DE375BBB04C7103D1CC1F0C026C4D8B3054469C503C5DEBB7EC5BF3D23402F45743D3E421D3F9B41833CA6BD45B819BDA7BF0D3203BDB0BF32C0533D56C011C144B963445944D1B7A93065B6433D22C596C0B9BB8C3CDBC1074677ADC83D07BFDD3A913F96B5CFC0C34406B0413EC942473FF2C1F14450428544CFBF9F43F7417F423742BB330A40534301BF953AC83CD4BC4EBECF3B4A45C8430DB8CD41D9B8B4AC3E408C423DC265BD77410DB1DFC04C4737363A43E8C17644A7C318426841F54486BEEE46B22CED37C0C09D4479B7E1B9783DA8BAF6C08E42CFC6463C"> : tensor<1x125xf16>
    %1 = stablehlo.constant dense<2.000000e+00> : tensor<1xf16>
    return %0, %1 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> tensor<1x125xf16> {
    %0 = stablehlo.constant dense<"0x20BD453DF4B0F14488C2B94224AB7DC108423E42DFC524C062381A31DE375BBB04C7103D1CC1F0C026C4D8B3054469C503C5DEBB7EC5BF3D23402F45743D3E421D3F9B41833CA6BD45B819BDA7BF0D3203BDB0BF32C0533D56C011C144B963445944D1B7A93065B6433D22C596C0B9BB8C3CDBC1074677ADC83D07BFDD3A913F96B5CFC0C34406B0413EC942473FF2C1F14450428544CFBF9F43F7417F423742BB330A40534301BF953AC83CD4BC4EBECF3B4A45C8430DB8CD41D9B8B4AC3E408C423DC265BD77410DB1DFC04C4737363A43E8C17644A7C318426841F54486BEEE46B22CED37C0C09D4479B7E1B9783DA8BAF6C08E42CFC6463C"> : tensor<1x125xf16>
    return %0 : tensor<1x125xf16>
  }
}

