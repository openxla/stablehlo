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
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<1x125xf16>, tensor<1xi64>, tensor<1xf16>) -> tensor<1x125xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x125xf16>, tensor<1x125xf16>) -> ()
    return %2 : tensor<1x125xf16>
  }
  func.func private @inputs() -> (tensor<1x125xf16> {mhlo.layout_mode = "default"}, tensor<1xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x1EB8E6BE6CBA2AC1A63BD0C5703EE3449C4051BB5CBD2D2883419144983E5F40784294C23CC5BBAEF73ED7B5B1C23E3A09C089C171C041BCE440DBB2F048E3412EC1893BC7C4A140D3341EC79E41EE3C95C4BA377DBDCD2FA9B33F3A4A46B9B73BC4F64036417941D5BA07446DC4C13F4544213DBAC0C83A4CC115B2A2BB71B92B3F66291CBACEBD253E36B92EC498C5E5B40EC2F53D52314F420236DC446C3F90452D3D55BA1137B1BDDCC5B93C8F437B3C76367DBA0845B0C6D6BAFC3C2EC1B345C5C092447F326D42ED45B346C841634506442738B0C10F454E37A23C0C3A8DB612C32BBC9140F6BD64AE61435443B2BE6940E2458ABC9141"> : tensor<1x125xf16>
    %cst_0 = stablehlo.constant dense<-3.029300e+00> : tensor<1xf16>
    return %cst, %cst_0 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> (tensor<1x125xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x1EB8E6BE6CBA2AC1A63BD0C5703EE3449C4051BB5CBD2D2883419144983E5F40784294C23CC5BBAEF73ED7B5B1C23E3A09C089C171C041BCE440DBB2F048E3412EC1893BC7C4A140D3341EC79E41EE3C95C4BA377DBDCD2FA9B33F3A4A46B9B73BC4F64036417941D5BA07446DC4C13F4544213DBAC0C83A4CC115B2A2BB71B92B3F66291CBACEBD253E36B92EC498C5E5B40EC2F53D52314F420236DC446C3F90452D3D55BA1137B1BDDCC5B93C8F437B3C76367DBA0845B0C6D6BAFC3C2EC1B345C5C092447F326D42ED45B346C841634506442738B0C10F454E37A23C0C3A8DB612C32BBC9140F6BD64AE61435443B2BE6940E2458ABC9141"> : tensor<1x125xf16>
    return %cst : tensor<1x125xf16>
  }
}
