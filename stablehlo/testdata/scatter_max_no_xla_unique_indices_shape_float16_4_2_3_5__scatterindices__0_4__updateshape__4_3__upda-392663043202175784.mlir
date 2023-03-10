// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>)
    %2 = call @expected() : () -> tensor<4x2x3x5xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xf16>, tensor<2xi32>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>) {
    %0 = stablehlo.constant dense<"0x142F9BC3C8BCD83B93C41E3D1C416FC5E4BD463DA7B58C42E1BC353771423DB933C47145DF32103319C3EB397C3DD7C093BCE63C3A3EF835BE3F9544CBB644BB2BC3A1C327BD70C07D3C55BE524047C3CF3DF1C28346874382339AC2E0C5493919C0F4C539BC6B3BF8B9AA40513C85C0224258C37F4242BACB4608C424BC254420C14C455D429C3C92C1A0C083BC063FF9401641FFAEBB4003B6493B4340E0C302C2D144F03E503B4446F5BF88C07B4378BC6CC02FBD2642C2437443444039C00544F93A2541AE1D884399C1E8B4A2322D40A83EED418E3287C198C1C542B0448DBFD9C63FC54F35F1BE09C0B9C424C4"> : tensor<4x2x3x5xf16>
    %1 = stablehlo.constant dense<[[2.982420e+00, -2.107420e+00, -2.271480e+00], [-1.498050e+00, -2.167970e+00, 1.381840e+00], [1.940430e+00, 4.167970e+00, 3.544920e-01], [1.336910e+00, -2.972660e+00, -5.283200e-01]]> : tensor<4x3xf16>
    return %0, %1 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xf16> {
    %0 = stablehlo.constant dense<"0x142F9BC3C8BCD83BF7411E3D1C416FC5E4BD463DA7B58C42E1BC353771423DB933C47145DF32103319C3EB397C3DD7C093BCE63C3A3EF835BE3F9544CBB644BB2BC3A1C327BD70C07D3C55BE524056C0CF3DF1C283468743873D9AC2E0C5493919C0F4C539BC6B3BF8B9AA40513C85C0224258C37F4242BACB4608C424BC2544C33F4C455D429C3C92C12B4483BC063FF9401641AC35BB4003B6493B4340E0C302C2D144F03E503B4446F5BF88C07B4378BC6CC02FBD2642C2437443444039C00544F93A2541AE1D884399C1E8B4A2322D40A83EED418E3287C198C1C542B0448DBFD9C63FC54F35F1BE09C0B9C424C4"> : tensor<4x2x3x5xf16>
    return %0 : tensor<4x2x3x5xf16>
  }
}

