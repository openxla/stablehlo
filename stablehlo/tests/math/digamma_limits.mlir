// RUN: stablehlo-opt --chlo-legalize-to-stablehlo %s | stablehlo-translate --interpret

func.func @main() -> (tensor<2xf32>, tensor<2xf64>) {
  %x32 = stablehlo.constant dense<[0.0, -0.0]> : tensor<2xf32>
  %x64 = stablehlo.constant dense<[0.0, -0.0]> : tensor<2xf64>
  %expected32 = stablehlo.constant dense<[0xFF800000, 0x7F800000]> : tensor<2xf32>
  %expected64 = stablehlo.constant dense<[0xFFF0000000000000, 0x7FF0000000000000]> : tensor<2xf64>
  %0 = chlo.digamma %x32 : tensor<2xf32> -> tensor<2xf32>
  %1 = chlo.digamma %x64 : tensor<2xf64> -> tensor<2xf64>
  check.expect_eq %0, %expected32 : tensor<2xf32>
  check.expect_eq %1, %expected64 : tensor<2xf64>
  return %0, %1 : tensor<2xf32>, tensor<2xf64>
}
