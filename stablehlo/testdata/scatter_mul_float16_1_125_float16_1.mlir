// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x125xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x125xf16>, tensor<1xf16>)
    %1 = call @expected() : () -> tensor<1x125xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<1x125xf16>, tensor<1xi64>, tensor<1xf16>) -> tensor<1x125xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x125xf16>, tensor<1x125xf16>) -> ()
    return %2 : tensor<1x125xf16>
  }
  func.func private @inputs() -> (tensor<1x125xf16> {mhlo.layout_mode = "default"}, tensor<1xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x974185BED5C4293EDF462B392A4503C27A3D20BFAC4479C671C1FAB56FC5A935B3BA40B945BDDA34074029AE3ABFB8BDDD31A1362EB4AEC63BBA1C379FB8843705C562C36BBEEFB8AC2C143D86BB5D4207C55839D2C024398C39494550401E3AD8B0BA40274478BC75BA553D14C3FCBCFCB4DEC0B5BEDA4266AFF2C7C731713D69B8F1C226BDE4C0A4BCDA40693F9E434A4429449940BF3E293E5D3E6FC54F3AF9BE2944CC40264570BAC3C001C754BD30C498BCDBB770BA6040F3357141C23CF13B9BB93CBE0ABC3A3EDB40AB415B39C144A43C5B3D1F322A2E6044E440A6BE5E40C63C9E3EBA3B7845D24374C55BBC0CC42344894394BEA03F"> : tensor<1x125xf16>
    %cst_0 = stablehlo.constant dense<2.445310e+00> : tensor<1xf16>
    return %cst, %cst_0 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> (tensor<1x125xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xD64685BED5C4293EDF462B392A4503C27A3D20BFAC4479C671C1FAB56FC5A935B3BA40B945BDDA34074029AE3ABFB8BDDD31A1362EB4AEC63BBA1C379FB8843705C562C36BBEEFB8AC2C143D86BB5D4207C55839D2C024398C39494550401E3AD8B0BA40274478BC75BA553D14C3FCBCFCB4DEC0B5BEDA4266AFF2C7C731713D69B8F1C226BDE4C0A4BCDA40693F9E434A4429449940BF3E293E5D3E6FC54F3AF9BE2944CC40264570BAC3C001C754BD30C498BCDBB770BA6040F3357141C23CF13B9BB93CBE0ABC3A3EDB40AB415B39C144A43C5B3D1F322A2E6044E440A6BE5E40C63C9E3EBA3B7845D24374C55BBC0CC42344894394BEA03F"> : tensor<1x125xf16>
    return %cst : tensor<1x125xf16>
  }
}
