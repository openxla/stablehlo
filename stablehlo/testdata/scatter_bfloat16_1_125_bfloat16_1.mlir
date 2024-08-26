// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x125xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x125xbf16>, tensor<1xbf16>)
    %1 = call @expected() : () -> tensor<1x125xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      stablehlo.return %arg1 : tensor<bf16>
    }) : (tensor<1x125xbf16>, tensor<1xi64>, tensor<1xbf16>) -> tensor<1x125xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> ()
    return %2 : tensor<1x125xbf16>
  }
  func.func private @inputs() -> (tensor<1x125xbf16> {mhlo.layout_mode = "default"}, tensor<1xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x2A3E3A4005BE0C40A4C01AC0FABF373F46BF56BF57C02EC00E3F354008BF40BF18404D4048BFBD3F4DC043402D40B63FADBE11BFFEBF174089C08EC06AC0B4BFB43D1F40B0402ABEFC3FEBBF57C0F1BE4240FE3EA23F8440984012BEDC3D66BF7B3F514069C077BFEE3DF43FC0BF903F9C3FA1C02640FA3FB54085BF11400FC02CBE73C0C53E1340B540DBBF66C075BB13C082C0C83F0AC01FBFF6BF22C0ACC0883E3A4051C08EC007C046BF31C08B3FFF3F5FC0CD3F5B3F8D3F683FB54033C0B6BF6CBFDD4064BE034039C0023EFBBFEC3F9CBF364099BCA83FCF3F024090BF2CC0323F8DBE44C03D3FA4C0A43E31C12C3F78C0C43F1CC09E40"> : tensor<1x125xbf16>
    %cst_0 = stablehlo.constant dense<7.734380e-01> : tensor<1xbf16>
    return %cst, %cst_0 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> (tensor<1x125xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x463F3A4005BE0C40A4C01AC0FABF373F46BF56BF57C02EC00E3F354008BF40BF18404D4048BFBD3F4DC043402D40B63FADBE11BFFEBF174089C08EC06AC0B4BFB43D1F40B0402ABEFC3FEBBF57C0F1BE4240FE3EA23F8440984012BEDC3D66BF7B3F514069C077BFEE3DF43FC0BF903F9C3FA1C02640FA3FB54085BF11400FC02CBE73C0C53E1340B540DBBF66C075BB13C082C0C83F0AC01FBFF6BF22C0ACC0883E3A4051C08EC007C046BF31C08B3FFF3F5FC0CD3F5B3F8D3F683FB54033C0B6BF6CBFDD4064BE034039C0023EFBBFEC3F9CBF364099BCA83FCF3F024090BF2CC0323F8DBE44C03D3FA4C0A43E31C12C3F78C0C43F1CC09E40"> : tensor<1x125xbf16>
    return %cst : tensor<1x125xbf16>
  }
}
