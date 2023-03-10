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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xbf16>, tensor<2xi32>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) {
    %0 = stablehlo.constant dense<"0xDD3F9D3F313E44C074BF61C0033FA7BF4540FBBBC1BFE83F2B40F63FA6C03340F93F4FBF1AC09CBE92C0B0BF25409E3D8BBE99C008BFB43F47C0753F4840983E79BF0140AC3F43C03340F43DECBF87C04DBF1940463EA4BF0ABF22C05E40963F82C0C43F0D40FA3F9340A23E66405FBE2E40AEC017C0DDBFE440C9407540D5BF7ABFB9BF61408DBF8CC09EBE933FAB3F0F412CBE754077C0C53F73C09E40B3BFC73E4DC03E4090C0C8C0B73FAA3D8FC04F408B3FF3BF0940F5BE083F0F40EC3EEDBFBEBF04C0A43FB34072C098C0C2C04DC04CBF62C0374040C0FA3FC9BD0ABF3EC0F83E85C0C5409BC06DC032C0293F"> : tensor<4x2x3x5xbf16>
    %1 = stablehlo.constant dense<[[-5.531250e+00, 1.726560e+00, 2.906250e+00], [-4.882810e-01, 3.593750e+00, 4.062500e+00], [-1.601560e+00, -8.375000e+00, 5.976560e-01], [8.945310e-01, 7.109380e-01, -1.375000e+00]]> : tensor<4x3xbf16>
    return %0, %1 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xbf16> {
    %0 = stablehlo.constant dense<"0xDD3F9D3F313E44C074BF61C0033FA7BF4540DD3FC1BFE83F2B40F63F3A403340F93F4FBF1AC09CBE92C0B0BF25409E3D8BBE99C008BFB43F47C0753F4840983E79BF0140AC3F43C03340F43DECBF66404DBF1940463EA4BF824022C05E40963F82C0C43F0D40FA3F9340A23E66405FBE2E40AEC017C0DDBFE440C9407540D5BF7ABFB9BF61408DBF8CC09EBE933FAB3F0F412CBE754077C0C53F73C09E40B3BFC73E4DC03E4090C0C8C0B73FAA3D8FC04F408B3FF3BF0940F5BE083F0F40EC3EEDBFBEBF04C0A43FB34072C098C0C2C0B0BF4CBF62C0374040C0FA3FC9BD0ABF3EC0F83E85C0C5409BC06DC032C0293F"> : tensor<4x2x3x5xbf16>
    return %0 : tensor<4x2x3x5xbf16>
  }
}

