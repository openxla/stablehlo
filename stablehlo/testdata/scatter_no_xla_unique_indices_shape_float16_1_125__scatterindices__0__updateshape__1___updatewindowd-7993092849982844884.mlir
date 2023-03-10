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
      stablehlo.return %arg1 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xf16>, tensor<1xi32>, tensor<1xf16>) -> tensor<1x125xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xf16>, tensor<1x125xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xf16>, tensor<1xf16>) {
    %0 = stablehlo.constant dense<"0xC13C6445ABBCF64195C28EAC09BDE13E9C4310C40834DC3F60BF1BC28AC02C34EB3D3D39DB3B3AB965C3794110BB374287C3BCBD1DBDCEC1CE452FBE1944E4417139F3254C34C5C4D73C99429A4372C47EC0B2B3EDC09838FABFC3462C9A1F4193C08D4184BF6DB5E6462AC41DC5D544A3C163370344E6BB98BE07B788BCE344E94426BE73B5C6440F3D1540B644EB422242AA3E75AF87420C3A6CC1ACBD004479C00D401D414B3E10BE94BE3EC0AD38E13CF0BC5F40C7AFB239692C8D3BD539BC40B3BE09440D3D0CC1FF3F31C44BC23BC01D44DB41EE3F13408445E6BA9FC07D3C55BFDE43B2BB6C3D76BDBB3D12BCEAB6723B0AC2F3BF3DC2"> : tensor<1x125xf16>
    %1 = stablehlo.constant dense<1.429690e+00> : tensor<1xf16>
    return %0, %1 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> tensor<1x125xf16> {
    %0 = stablehlo.constant dense<"0xB83D6445ABBCF64195C28EAC09BDE13E9C4310C40834DC3F60BF1BC28AC02C34EB3D3D39DB3B3AB965C3794110BB374287C3BCBD1DBDCEC1CE452FBE1944E4417139F3254C34C5C4D73C99429A4372C47EC0B2B3EDC09838FABFC3462C9A1F4193C08D4184BF6DB5E6462AC41DC5D544A3C163370344E6BB98BE07B788BCE344E94426BE73B5C6440F3D1540B644EB422242AA3E75AF87420C3A6CC1ACBD004479C00D401D414B3E10BE94BE3EC0AD38E13CF0BC5F40C7AFB239692C8D3BD539BC40B3BE09440D3D0CC1FF3F31C44BC23BC01D44DB41EE3F13408445E6BA9FC07D3C55BFDE43B2BB6C3D76BDBB3D12BCEAB6723B0AC2F3BF3DC2"> : tensor<1x125xf16>
    return %0 : tensor<1x125xf16>
  }
}

