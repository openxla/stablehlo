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
      stablehlo.return %arg1 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xbf16>, tensor<2xi32>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) {
    %0 = stablehlo.constant dense<"0x80C0343DA43FBE3E3CC0C6406C3F734018407CC0913FC8C0F43F104033BF553FCFC076BF41BFA6BE16402A40F5C081BFC9BDDEBE0940093F1FC088404CC0383E0E40D5405DBFBABF4F406CBF50BF58C0F1BF903FA5BE88C0DEBE573F09C019C0A840A83FFBBF9F3F0F40804086BF0DC18040933F52BF834004402B40C9BFB240FEBE40C0ED3F00C0573F4140A83F0DC02DC0A13E2740C2BFE33D5DC0F73F4E3D7EBE283FD4C09C3E52C0583E0340B1BF95BBAEBEC0BFE83E0540CC3F6140A33FCE402AC0A23FE63F9D3F23BFCE3F333FD63F85401C40863EA63F9340E6BFDFBF60BFF0C0444091C090404DC071BE57C0"> : tensor<4x2x3x5xbf16>
    %1 = stablehlo.constant dense<[[-5.875000e+00, -3.015630e+00, -8.242180e-01], [3.250000e+00, 3.140630e+00, -2.359380e+00], [-4.437500e+00, -3.531250e+00, 4.437500e+00], [8.515620e-01, 5.531250e+00, -1.718750e+00]]> : tensor<4x3xbf16>
    return %0, %1 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xbf16> {
    %0 = stablehlo.constant dense<"0x80C0343DA43FBE3EBCC0C6406C3F7340184041C0913FC8C0F43F104053BF553FCFC076BF41BFA6BE16402A40F5C081BFC9BDDEBE0940093F1FC088404CC0383E0E40D5405040BABF4F406CBF50BF4940F1BF903FA5BE88C017C0573F09C019C0A840A83FFBBF9F3F0F40804086BF0DC18040933F52BF834004402B40C9BFB2408EC040C0ED3F00C0573F62C0A83F0DC02DC0A13E8E40C2BFE33D5DC0F73F4E3D7EBE283FD4C09C3E52C0583E0340B1BF95BBAEBEC0BFE83E0540CC3F5A3FA33FCE402AC0A23FB1409D3F23BFCE3F333FDCBF85401C40863EA63F9340E6BFDFBF60BFF0C0444091C090404DC071BE57C0"> : tensor<4x2x3x5xbf16>
    return %0 : tensor<4x2x3x5xbf16>
  }
}

