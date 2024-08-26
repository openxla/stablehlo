// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>)
    %1 = call @expected() : () -> tensor<4x2x3x5xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<4x2x3x5xbf16>, tensor<2xi64>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> ()
    return %2 : tensor<4x2x3x5xbf16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16> {mhlo.layout_mode = "default"}, tensor<4x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x6440CEBE344086401BC082C027C0913FC3403440753EF83F7240DB3F8DBF23C06C40CE3F13C08540F1BF42404FC098BF1B40083E72BF81BFA63DFF40F5BF274071C04AC0A5C07EC044C0A13F793F2640A73FB43F12C04CBF443FF13FD8BF1840BC3F624060C007C056BF46BF4D3FA13F31BF6CC0CBBF13BDBE40B6C05DC07240CBBF6040F83FADC09D3F7C4026C0FCBD55BF70BDE1C083C02B4026C0AF40954027C0F03E354077407A40123F88401D409AC041C0FA3FB340D6408DC0D2BF02C0ADC0883BAEBF703FF33F0A4015407C3FB3401ABF7240D1BFC53F3140D9BF66C05BBECBBEAF3F49C00FC01B4043409F3F"> : tensor<4x2x3x5xbf16>
    %cst_0 = stablehlo.constant dense<[[7.968750e-01, 3.687500e+00, -8.437500e-01], [1.257810e+00, 8.007810e-01, -2.125000e+00], [-9.531250e-01, -1.110840e-02, 1.007810e+00], [1.968750e+00, -8.437500e-01, -6.875000e+00]]> : tensor<4x3xbf16>
    return %cst, %cst_0 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x6440CEBE344086404C3F82C027C0913FC3406C40753EF83F7240DB3F58BF23C06C40CE3F13C08540F1BF42404FC098BF1B40083E72BF81BFA63DFF40F5BF274071C04AC0A13F7EC044C0A13F793F2640A73FB43F12C04CBF443FF13FD8BF1840BC3F624060C007C056BF46BF4D3FA13F31BF6CC0CBBF13BDBE40B6C05DC0724074BF6040F83FADC09D3F7C4026C0FCBD55BF70BD813F83C02B4026C0AF40954027C0F03E354077407A40123F88401D409AC041C0FA3FB340D6408DC0FC3F02C0ADC0883BAEBF703FF33F0A4015407C3FB3401ABF7240D1BFC53F3140D9BF66C05BBECBBEAF3F49C00FC01B4043409F3F"> : tensor<4x2x3x5xbf16>
    return %cst : tensor<4x2x3x5xbf16>
  }
}
