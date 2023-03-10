// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xf16>, tensor<1x3xf16>)
    %2 = call @expected() : () -> tensor<1x50x3xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      stablehlo.return %arg1 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xf16>, tensor<1xi32>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16>, tensor<1x3xf16>) {
    %0 = stablehlo.constant dense<"0x4FBDB04007B45F4374C30BC4F84567BD17BBC7BE5AC16F366030DAC142C4033CBBBF67C1983DA0BC573B173532433CB69D3D0CB082BC9BC4EFBAF2C43EB4EEC0D9BEC9C1403D47360E4020C29BB65CC5D6458BC227457F4561C227B5A244D7C0954446BD903EBEC67A3A87BCE13FC64340B389BD65412B406041F63F3AC657398BC2A0BF193FB5C0BABFC041D8BF1DC59E432B379F417DBE54C501BFBBB747C060BF64B21740B2416FBBFD4140BEF6395FBC50C5B04560435E45B6C64BBFF2C22146613D7B452B41C54521C2A23F4ABD9B243DC5ACB5953262BA98B949BCF2445E449D4022C54144103849C29544FE4189C16AC28042BB41F0444C45D6BCE93ED6C592480B4408449E381FBA4BB881C297C343C473C304C0663C5EC555C5D43BA1C102417FC1E03F95C182BF"> : tensor<1x50x3xf16>
    %1 = stablehlo.constant dense<[[-6.240230e-01, -2.066410e+00, -1.768550e+00]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x50x3xf16> {
    %0 = stablehlo.constant dense<"0x4FBDB04007B45F4374C30BC4F84567BD17BBC7BE5AC16F366030DAC142C4033CBBBF67C1983DA0BC573B173532433CB69D3D0CB082BC9BC4EFBAF2C43EB4EEC0D9BEC9C1403D47360E4020C29BB65CC5D6458BC227457F4561C227B5A244D7C0954446BD903EBEC67A3A87BCE13FC64340B389BD65412B406041F63F3AC657398BC2A0BF193FB5C0BABFC041D8BF1DC59E432B379F417DBE54C501BFBBB747C060BF64B21740B2416FBBFD4140BEF6395FBC50C5B04560435E45B6C64BBFF2C2FEB822C013BF2B41C54521C2A23F4ABD9B243DC5ACB5953262BA98B949BCF2445E449D4022C54144103849C29544FE4189C16AC28042BB41F0444C45D6BCE93ED6C592480B4408449E381FBA4BB881C297C343C473C304C0663C5EC555C5D43BA1C102417FC1E03F95C182BF"> : tensor<1x50x3xf16>
    return %0 : tensor<1x50x3xf16>
  }
}

