// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>)
    %2 = call @expected() : () -> tensor<4x2x3x5xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xbf16>, tensor<2xi32>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) {
    %0 = stablehlo.constant dense<"0x164000C052BFE3407E40F940F7BF55BF0F40243E49C10040EBBF3AC0B4BF2FC07540EA3D2FC093BF1DBF2140B63FD43FE8BFEBBFDCC0BF3E6640B6BE5EC0C64095BF4DBF0A3F27BFDCBFF13F84C00CC14EBFBF40F8BF99C0443F61BD61C0C44015C0D3BFD2C08F3F933FA4BF863E4E3F66C089BF92C0E9BC12C00A3F95C02CC0DBC021C0183FD0BFD63EADBF33C04D4000402140B93FA0C0BF3F1440A8C0A03F103FEDBF8E40123FEBBD1CC0CB3FC13FF43FC4BFCA3E394009BE2D3F5C3FFBBF52BF90C0CCBF753D563F48C089BD293F5EC0214060C0993FD24004C0EEBFAFC0DFBFA83F5240D53E163FAC40BE3E703F"> : tensor<4x2x3x5xbf16>
    %1 = stablehlo.constant dense<[[-1.921880e+00, 4.707030e-01, -2.796880e+00], [-1.210940e+00, 1.343750e+00, -7.187500e+00], [-1.578130e+00, -5.546880e-01, -1.906250e+00], [2.562500e+00, -4.781250e+00, -2.843750e+00]]> : tensor<4x3xbf16>
    return %0, %1 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xbf16> {
    %0 = stablehlo.constant dense<"0x164000C052BFE340F6BFF940F7BF55BF0F40243E49C10040EBBF3AC033C02FC07540EA3D2FC093BF1DBF2140B63FD43FE8BFEBBFDCC0BF3E6640B6BE5EC0C64095BF4DBF9BBF27BFDCBFF13F84C00CC14EBFBF40F8BF99C0E6C061BD61C0C44015C0D3BFD2C08F3F933FA4BF863E4E3F66C089BF92C0E9BC12C00A3F95C02CC0DBC021C0183FD0BFD63EADBF33C04D4000402140F4BFA0C0BF3F1440A8C0A03F103FEDBF8E40123FEBBD1CC0CB3FC13FF43FC4BFCA3E394009BE2D3F5C3FFBBF52BF90C0CCBF99C0563F48C089BD293F5EC0214060C0993FD24004C0EEBFAFC0DFBFA83F5240D53E163FAC40BE3E703F"> : tensor<4x2x3x5xbf16>
    return %0 : tensor<4x2x3x5xbf16>
  }
}

