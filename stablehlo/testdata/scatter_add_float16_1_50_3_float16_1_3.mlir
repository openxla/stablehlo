// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x50x3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<32> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x50x3xf16>, tensor<1x3xf16>)
    %1 = call @expected() : () -> tensor<1x50x3xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<1x50x3xf16>, tensor<1xi64>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> ()
    return %2 : tensor<1x50x3xf16>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16> {mhlo.layout_mode = "default"}, tensor<1x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x26BC103799B822411FC71D42A1AFCBC52DBD0B3DA03C5DB49FB2EA31F344A3C643C72A42633AF33C34454ABD5D2F653AD24100C4CDB8CA341EC1A5C47344C4C1F03AC0C502C0A8B131BB6DC11CC6BBB440BFE03ABC322632F142714130428542D0C18ABB93BE6CBBD5BFBEAE6C400BC4F0BB71C434C0C13BF9C1DC3864C1F7C1ACC63046204037418F44AF35C33D8634ADBD3DB151436FBEBFC6393D99C02F3E1DBF743CA04171423541B3A541C63D3C6E3D9C3E193C5142C2BCE9C4BF3C2B4014BC2244F6BFD4C4FB44F9B71EBB3DB808489F421F40733F552D64C42D3E64B0F83E2F34E83D3E399040F437EB3D9745D9C093440C418E381EBE8441D2429F39A33C13BD183B2E476BC0C53FCB385A3EB63CD12E7EB80C45A1406442504175C636BAB03DC643574199BD3444"> : tensor<1x50x3xf16>
    %cst_0 = stablehlo.constant dense<[[8.076170e-01, 2.814450e+00, 1.016600e+00]]> : tensor<1x3xf16>
    return %cst, %cst_0 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> (tensor<1x50x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x26BC103799B822411FC71D42A1AFCBC52DBD0B3DA03C5DB49FB2EA31F344A3C643C72A42633AF33C34454ABD5D2F653AD24100C4CDB8CA341EC1A5C47344C4C1F03AC0C502C0A8B131BB6DC11CC6BBB440BFE03ABC322632F142714130428542D0C18ABB93BE6CBBD5BFBEAE6C400BC4F0BB71C434C0C13BF9C1DC3864C1F7C1ACC63046204037418F44AF35C33D8634ADBD3DB151436FBEBFC6393D99C02F3E1DBF743CA04171423541B3A541C63D3C6E3D9C3E193C5142C2BCE9C4BF3C2B40C8B2F246CABBD4C4FB44F9B71EBB3DB808489F421F40733F552D64C42D3E64B0F83E2F34E83D3E399040F437EB3D9745D9C093440C418E381EBE8441D2429F39A33C13BD183B2E476BC0C53FCB385A3EB63CD12E7EB80C45A1406442504175C636BAB03DC643574199BD3444"> : tensor<1x50x3xf16>
    return %cst : tensor<1x50x3xf16>
  }
}
