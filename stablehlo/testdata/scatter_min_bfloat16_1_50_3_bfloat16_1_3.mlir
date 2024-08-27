// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x50x3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<32> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>)
    %1 = call @expected() : () -> tensor<1x50x3xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<1x50x3xbf16>, tensor<1xi64>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> ()
    return %2 : tensor<1x50x3xbf16>
  }
  func.func private @inputs() -> (tensor<1x50x3xbf16> {mhlo.layout_mode = "default"}, tensor<1x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x54BF50BEC53FF9BFA03D5B3FF03FDABFA23FEB3F48C0554050C01B3FD8BEB8BD403F2DBF23BEC140A93D8CBF343F8A407EBF823FB1BF073D40401CBF84C08C3D24BC593FC0C0C5C0653F35C08EC0F63F583F30C060BEC6BF58C041C0134013C09A3FC53FD83F82BF92BE23C0F53F0A3F88408CC0FF403BBFCCBF014079C0ABBEB540F53FFABF24BD50407BBE893F0FC0423E04409B3F28C0C8BEAC40854089BF6EC0A7BFAE406B40CC3FAF401FC0203FE9BDE140E1BF7B40E2BF9E3F9540AD40BB3F23BFC13FA14013BF164014405AC094BE393FD83FF9BE28402DBF85C065C056404F3F7C4075C081BF893F14C0304013407A4003BFC03F1540874019BFD3C08AC09EBF0FC033BFB6BF4E4046BF073EB44072BF79C00CC0F23FA9BFDCBE3DC02540B1C084C0A33F49401340"> : tensor<1x50x3xbf16>
    %cst_0 = stablehlo.constant dense<[[1.513670e-01, -3.015630e+00, -3.812500e+00]]> : tensor<1x3xbf16>
    return %cst, %cst_0 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> (tensor<1x50x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x54BF50BEC53FF9BFA03D5B3FF03FDABFA23FEB3F48C0554050C01B3FD8BEB8BD403F2DBF23BEC140A93D8CBF343F8A407EBF823FB1BF073D40401CBF84C08C3D24BC593FC0C0C5C0653F35C08EC0F63F583F30C060BEC6BF58C041C0134013C09A3FC53FD83F82BF92BE23C0F53F0A3F88408CC0FF403BBFCCBF014079C0ABBEB540F53FFABF24BD50407BBE893F0FC0423E04409B3F28C0C8BEAC40854089BF6EC0A7BFAE406B40CC3FAF401FC0203FE9BDE140E1BF7B40E2BF9E3F9540AD401B3E41C074C0A14013BF164014405AC094BE393FD83FF9BE28402DBF85C065C056404F3F7C4075C081BF893F14C0304013407A4003BFC03F1540874019BFD3C08AC09EBF0FC033BFB6BF4E4046BF073EB44072BF79C00CC0F23FA9BFDCBE3DC02540B1C084C0A33F49401340"> : tensor<1x50x3xbf16>
    return %cst : tensor<1x50x3xbf16>
  }
}
