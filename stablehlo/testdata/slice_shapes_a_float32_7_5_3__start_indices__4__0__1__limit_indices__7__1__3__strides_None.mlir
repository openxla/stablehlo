// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<7x5x3xf32>
    %1 = call @expected() : () -> tensor<3x1x2xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<[7, 1, 3]> : tensor<3xi64>, start_indices = dense<[4, 0, 1]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<7x5x3xf32>) -> tensor<3x1x2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x1x2xf32>, tensor<3x1x2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<7x5x3xf32> {
    %0 = stablehlo.constant dense<"0x81316BBF0B8B45BF0408CABEBA75533D4DF10EBF5D1E10C036D2D3BF417D32BF38EEB2BED2F4CB3ED1533BC0C3C0D0BFF138B740286463BFBCA932C0CA5C1AC000AF2F401CCA4C3F90DD4CBF1D3FF4BFF6F5FBC0BFE9BDC0C0D0A2C06C0486BFDDC9D33F0B7D823F4D17A5BFCBA80040544CD0406F5994400FFE6D3F81D690BF1856AB3EE337A13F45A2A6C000812C40797E9C3F2488AC4068C98BBF0CC127C02F5E53409387D540099DDC3F02AD624024B4D340739EB63ED57AE03D011E624050F010C0290B5840186FDFBFAD61F43FC573B1C061A3F840D749B73F4353EF3FCE5E5E3EA564213FD96ED83F8DD634BF89F31440D3A927BFADB9113F0C79D0BFE74E63BE7789963FE4348DC093DEE13F60C526C0599988C0BEB2C93FE9CE13C07DFD1F3BA9699D403B4303C014D734401FCE3E402639FA3EB8E59C3FC9D612C087C23D4012119DBFFA863F3FA552CB3ECC11D2BF13D23AC0C7545E3F8488B43FB104BDBF60301540A5592340A497B13F7248F63F5A930AC0561FA0C0DDC5C7BE870FA43F49743A40C81D1540B1B6BCBFA59B3C3FD71B47C00BFF4AC0289B4640418FC23E"> : tensor<7x5x3xf32>
    return %0 : tensor<7x5x3xf32>
  }
  func.func private @expected() -> tensor<3x1x2xf32> {
    %0 = stablehlo.constant dense<[[[-0.654935061, 0.569239438]], [[2.98133063, 0.488717258]], [[1.3874402, 1.92408586]]]> : tensor<3x1x2xf32>
    return %0 : tensor<3x1x2xf32>
  }
}
