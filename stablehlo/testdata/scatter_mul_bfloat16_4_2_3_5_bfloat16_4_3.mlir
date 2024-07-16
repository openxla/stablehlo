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
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<4x2x3x5xbf16>, tensor<2xi64>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> ()
    return %2 : tensor<4x2x3x5xbf16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16> {mhlo.layout_mode = "default"}, tensor<4x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xB9C0D8BF4B40183F3AC0403F6E3F024088C06140DD3E7F4016BF593FC9BFB7403FC02240B5BE4E4056BE83C0F13FCCBE57BFEBBF933FBF40853F80405BC0D23F9CBF55409ABFB140EC3FB73E824056C0AC3E05C0863F3CC01740CC3F2BC06AC09CC02BBFEDBF6540D83F9FC07B407C3F45BF823F0E4050406140DEC006C039C0CAC09C3F05BE62BF8C3D774052C027BECF404FC0B2BE0941823E50C08AC0204018BFAEBF963F16C0A9C0F43FD73F8F3D8DC08340D23FA73E9E40A7C04BC045C084406D3F12C0184048C09BC0F7BEA03FBDBF36C0AFC0F33C3EBFED3E27C02040D3C0A8C08FC014C04ABF823F0940E5BF"> : tensor<4x2x3x5xbf16>
    %cst_0 = stablehlo.constant dense<[[-6.062500e+00, -1.445310e+00, 4.687500e+00], [3.203130e+00, 6.289060e-01, 2.328130e+00], [3.687500e+00, -8.320310e-01, 2.500000e+00], [-3.187500e+00, -3.437500e-01, -3.500000e+00]]> : tensor<4x3xbf16>
    return %cst, %cst_0 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xB9C0D8BF4B40183F8D41403F6E3F024088C0A3C0DD3E7F4016BF593FECC0B7403FC02240B5BE4E4056BE83C0F13FCCBE57BFEBBF933FBF40853F80405BC0D23F9CBF554077C0B140EC3FB73E824007C0AC3E05C0863F3CC0B040CC3F2BC06AC09CC02BBFEDBF6540D83F9FC07B407C3F45BF823F0E4050406140DEC006C039C0BAC19C3F05BE62BF8C3D4EC052C027BECF404FC05EBF0941823E50C08AC0204018BFAEBF963F16C0A9C0F43FD73F8F3D8DC08340D23FA73E9E40A7C0224145C084406D3F12C051BF48C09BC0F7BEA03FA54036C0AFC0F33C3EBFED3E27C02040D3C0A8C08FC014C04ABF823F0940E5BF"> : tensor<4x2x3x5xbf16>
    return %cst : tensor<4x2x3x5xbf16>
  }
}
