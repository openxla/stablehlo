// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>)
    %1 = call @expected() : () -> tensor<4x2x3x5xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      stablehlo.return %arg1 : tensor<f16>
    }) : (tensor<4x2x3x5xf16>, tensor<2xi64>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> ()
    return %2 : tensor<4x2x3x5xf16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf16> {mhlo.layout_mode = "default"}, tensor<4x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x4B44F934C0B84FBB1BB215C100C141405C3F18C45F4469C21841ED4477BEBEB629BE5F3D88C5F939FBC4D943013723C17740E44437C2DB39DBBD29C460BDCD3F6BB6303E89C2A344403C61BCE2B8713E17B05DBE653C523E873E94BF50C5F921D2C10C46ED322C4557C550B421BDCBC3FE4578B68DBCB3BC43C397B94B440042714383C2423E4B437C40B038C7B9AE372C444EC536BF8A41702CC2B7DCBCDB40C04503444D31283034C321BFD736F2B1EB3E76414D3DBEC33B433EBF3C40C742A1C2D7437F3566BC14C5B740173829C10D43C23826BDD5C380C1CCC002C1D24030BEA63C2B3407BCA547D0403A3C2542"> : tensor<4x2x3x5xf16>
    %cst_0 = stablehlo.constant dense<[[-3.244140e+00, -4.222660e+00, 1.871090e+00], [-1.164060e+00, 7.133780e-01, -5.019530e-01], [-1.629880e+00, 8.833000e-01, 3.818360e+00], [-1.457030e+00, 8.632810e+00, -4.613280e+00]]> : tensor<4x3xf16>
    return %cst, %cst_0 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x4B44F934C0B84FBB7DC215C100C141405C3F39C45F4469C21841ED447C3FBEB629BE5F3D88C5F939FBC4D943013723C17740E44437C2DB39DBBD29C460BDCD3F6BB6303EA8BCA344403C61BCE2B8B53917B05DBE653C523E04B894BF50C5F921D2C10C46ED322C4557C550B421BDCBC3FE4578B68DBCB3BC43C397B94B44004285BE83C2423E4B437C40113BC7B9AE372C444EC5A3438A41702CC2B7DCBCDB40C04503444D31283034C321BFD736F2B1EB3E76414D3DBEC33B433EBFD4BDC742A1C2D7437F35514814C5B740173829C19DC4C23826BDD5C380C1CCC002C1D24030BEA63C2B3407BCA547D0403A3C2542"> : tensor<4x2x3x5xf16>
    return %cst : tensor<4x2x3x5xf16>
  }
}
