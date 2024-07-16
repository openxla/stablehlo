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
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<4x2x3x5xf16>, tensor<2xi64>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> ()
    return %2 : tensor<4x2x3x5xf16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf16> {mhlo.layout_mode = "default"}, tensor<4x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x8143E842EA381146254530C2C643D5C38F427FC023C080C0963D33C198C2693715BC283F9DC4D8393FB958BF78462E39BFBE983854C435C58042783FCCC08AC5F04277442D3CABB4B4C442C3164608B4BEC48A440CBCF4428BB811C6DAB96245A8BCE83BF3BA4CB02AC49FC27DC4A844B62FB7C06645424172369DB1873DDD43932A6841A1414B41A64466C4613F41B5FDBA89B89941653C55BD65BC88C0BBAA07BF47BAE342F0BF403E3E3FB5C50BC5403EDA3D31400EBBAFBDA5384EBB15BD823B8BA9E93FF9C0403E1CC1243CCA4739B8FFC40A332F4181391B3C2B419138B0485139D2C5F2C2A63BCE44A1C6D2BB"> : tensor<4x2x3x5xf16>
    %cst_0 = stablehlo.constant dense<[[1.444090e-01, 3.208980e+00, -1.485350e+00], [-2.306640e+00, -1.086910e+00, 2.781250e+00], [2.869140e+00, 2.907710e-01, -1.608400e+00], [5.746090e+00, -1.758790e+00, -2.523440e+00]]> : tensor<4x3xf16>
    return %cst, %cst_0 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x8143E842EA381146F13930C2C643D5C38F4237C723C080C0963D33C1E644693715BC283F9DC4D8393FB958BF78462E39BFBE983854C435C58042783FCCC08AC5F0427744D1C0ABB4B4C442C316466234BEC48A440CBCF44251BE11C6DAB96245A8BCE83BF3BA4CB02AC49FC27DC4A844B62FB7C06645424172369DB1873DDD43B7306841A1414B41A6441EBD613F41B5FDBA89B880C4653C55BD65BC88C0BBAA07BF47BAE342F0BF403E3E3FB5C50BC5403EDA3D31400EBBAFBDA5383FC515BD823B8BA9E93F5F44403E1CC1243CCA47543DFFC40A332F4181391B3C2B419138B0485139D2C5F2C2A63BCE44A1C6D2BB"> : tensor<4x2x3x5xf16>
    return %cst : tensor<4x2x3x5xf16>
  }
}
