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
      stablehlo.return %arg1 : tensor<f16>
    }) : (tensor<1x125xf16>, tensor<1xi64>, tensor<1xf16>) -> tensor<1x125xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x125xf16>, tensor<1x125xf16>) -> ()
    return %2 : tensor<1x125xf16>
  }
  func.func private @inputs() -> (tensor<1x125xf16> {mhlo.layout_mode = "default"}, tensor<1xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xB2B69744D03B57C0F9BC07B4B5C5BF45D8C2F3C015B854BDDDB8D145E3405141653173C570C71D427EBFD641EEBCD243D1C4B3BF4A43733FD7438741B63C313E87B698C382449FBE0CC635419D37D8C10E31CE407444E5421C42E7C1BAC529416FC4B2B870C39C3EDFC36A34E13850C4FE4380B8B6BD3AC445427DC04C3986C50CBBA4BCBCB9F5C1163DB3C493BCD1C311C2B64130C5A14058BA2FBDD3BA78AA9D3CAE3BA4BD73C6B0C04C40D2C1BB3D2FC53840663A5441973FD93282C7EBC0BD44AAB29243F644AABDC13BDD34CFC345BE37437AC1B2C186405444B53CD4C0AF3D683D9D471F451544ADBE873D90BFBB23DBBE45C10DBC39B8"> : tensor<1x125xf16>
    %cst_0 = stablehlo.constant dense<-2.001950e-01> : tensor<1xf16>
    return %cst, %cst_0 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> (tensor<1x125xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x68B29744D03B57C0F9BC07B4B5C5BF45D8C2F3C015B854BDDDB8D145E3405141653173C570C71D427EBFD641EEBCD243D1C4B3BF4A43733FD7438741B63C313E87B698C382449FBE0CC635419D37D8C10E31CE407444E5421C42E7C1BAC529416FC4B2B870C39C3EDFC36A34E13850C4FE4380B8B6BD3AC445427DC04C3986C50CBBA4BCBCB9F5C1163DB3C493BCD1C311C2B64130C5A14058BA2FBDD3BA78AA9D3CAE3BA4BD73C6B0C04C40D2C1BB3D2FC53840663A5441973FD93282C7EBC0BD44AAB29243F644AABDC13BDD34CFC345BE37437AC1B2C186405444B53CD4C0AF3D683D9D471F451544ADBE873D90BFBB23DBBE45C10DBC39B8"> : tensor<1x125xf16>
    return %cst : tensor<1x125xf16>
  }
}
