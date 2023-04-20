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
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xbf16>, tensor<2xi32>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) {
    %0 = stablehlo.constant dense<"0xC5C0BF3F7CC078BFC9C0F33FC43F40BF04C0454020BEA7C01CC09FBF1ABF8D3E3140374007405940C7C08E404FBF294049C05A4075C00AC066408F4059C014C0854009BF0E3F503F893C09C02DC0D83FEABF6C3C1AC0B5C08E4059C043408BC00140E7BE14C0BEBE77BF00408C3EF23EA1406A40E8BE48409B40364000C00D4043C015C00A3E0440D140C440E43F40C0F4BF9EC06CC0A8C099403D403DC0C9C018C0BEBEE43E2EC00CC082BC933E56BEEC3F16C0C73FF4C0713E30C010C090BF4740A73E06407A4047C0BABFA9C0E7BE98BFB7BE9E4007401B3E5740383D3440A84051C0563E02BE96BE9640943F2EBF"> : tensor<4x2x3x5xbf16>
    %1 = stablehlo.constant dense<[[-2.875000e+00, 2.937500e+00, 4.750000e+00], [-3.250000e+00, 1.943360e-01, 5.859380e-01], [2.609380e+00, -5.937500e-01, -6.250000e-01], [3.906250e+00, 1.039060e+00, 6.187500e+00]]> : tensor<4x3xbf16>
    return %0, %1 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xbf16> {
    %0 = stablehlo.constant dense<"0xC5C0BF3F7CC078BF12C1F33FC43F40BF04C0C04020BEA7C01CC09FBF85408D3E3140374007405940C7C08E404FBF294049C05A4075C00AC066408F4059C014C0854009BF2CC0503F893C09C02DC0F13FEABF6C3C1AC0B5C0A14059C043408BC00140E7BE14C0BEBE77BF00408C3EF23EA1406A40E8BE48409B40364000C00D40E0BE15C00A3E0440D140B140E43F40C0F4BF9EC08AC0A8C099403D403DC0C9C018C0BEBEE43E2EC00CC082BC933E56BEEC3F16C0C73FF4C0713E30C0D43F90BF4740A73E06409E4047C0BABFA9C0E7BEA040B7BE9E4007401B3E5740383D3440A84051C0563E02BE96BE9640943F2EBF"> : tensor<4x2x3x5xbf16>
    return %0 : tensor<4x2x3x5xbf16>
  }
}

