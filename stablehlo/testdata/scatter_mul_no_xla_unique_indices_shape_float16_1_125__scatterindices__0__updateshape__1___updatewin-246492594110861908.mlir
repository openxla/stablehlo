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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xf16>, tensor<1xi32>, tensor<1xf16>) -> tensor<1x125xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xf16>, tensor<1x125xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xf16>, tensor<1xf16>) {
    %0 = stablehlo.constant dense<"0x1E3ABBB9CE406DC2EFC36A3F3F412FB5C5C44135C7C2CEBDF74589C1B83ED130E73A91AB9DC110C12840EDBD37C2E139CAC0A24619C214C10040352EC3B86F443AB1BCB80DB9A44500346BB9B0375844E8405F402ABC9040763867C554C4EF40B6446F458044BB3FAFB461C67C40B7C5783ADCB7A7432AC56040B843AC3C6A3C1AB0CEC4A3C17D3FB7315E41A0C09D3FC24172C5F1C07FC19A3B6533BF3502C15441C03DA6BD624149B446C536BD1A42BFBE9A9A8A3A56412EBD284442BB72BF0EC4DF93B7BFF936CEC1383A37C2E641F74350BC9FC356BB9AB909C2D73CC1C2A8BC34C158C4DEBDA6C4834412C72B404938E7B93FBC3D344944"> : tensor<1x125xf16>
    %1 = stablehlo.constant dense<-3.693360e+00> : tensor<1xf16>
    return %0, %1 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> tensor<1x125xf16> {
    %0 = stablehlo.constant dense<"0xA6C1BBB9CE406DC2EFC36A3F3F412FB5C5C44135C7C2CEBDF74589C1B83ED130E73A91AB9DC110C12840EDBD37C2E139CAC0A24619C214C10040352EC3B86F443AB1BCB80DB9A44500346BB9B0375844E8405F402ABC9040763867C554C4EF40B6446F458044BB3FAFB461C67C40B7C5783ADCB7A7432AC56040B843AC3C6A3C1AB0CEC4A3C17D3FB7315E41A0C09D3FC24172C5F1C07FC19A3B6533BF3502C15441C03DA6BD624149B446C536BD1A42BFBE9A9A8A3A56412EBD284442BB72BF0EC4DF93B7BFF936CEC1383A37C2E641F74350BC9FC356BB9AB909C2D73CC1C2A8BC34C158C4DEBDA6C4834412C72B404938E7B93FBC3D344944"> : tensor<1x125xf16>
    return %0 : tensor<1x125xf16>
  }
}

