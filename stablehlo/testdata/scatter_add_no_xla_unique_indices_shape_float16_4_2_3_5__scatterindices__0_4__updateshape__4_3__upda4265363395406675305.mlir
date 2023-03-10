// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>)
    %2 = call @expected() : () -> tensor<4x2x3x5xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xf16>, tensor<2xi32>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>) {
    %0 = stablehlo.constant dense<"0x7CC46B440E449EC435B7DC3A39411BBEA8C3DC3B89409A4487C24CBC6E25D23C75C0934064448445A53D14BD14435E3E5441DEB91A4243B45544A746023AB0C195C1A6393DC095C2D0C0043557C5713E2E458DC2A23C883A32BB16BCA4BF1BC4FD3CF73AF23A42B4F0454FBF9526B24336C41E3DBABDA6BFC33A57BE383D2045633C33C863B4E5ACC24422403E381DC0CF3B28C259C448C221C6ECBCC14230441835A74380444A3DA2394C40E63C00BE01B9BAC19AC4313DF6B9DA4666C430B6E6C4E8C5C3BB703C58C7C342DD3F384139BCB13FE7B942BC1D3F42BE13C5F1C0D5B0DD3C3A3C193C7A40EC456F3880C6"> : tensor<4x2x3x5xf16>
    %1 = stablehlo.constant dense<[[1.154330e-02, 4.171880e+00, -1.665040e+00], [3.681640e+00, 1.291020e+00, -1.731450e+00], [6.066400e+00, -3.140630e+00, 7.402340e-01], [-3.626950e+00, -7.314450e-01, -2.468750e+00]]> : tensor<4x3xf16>
    return %0, %1 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xf16> {
    %0 = stablehlo.constant dense<"0x7CC46B440E449EC406B7DC3A39411BBEA8C3284589409A4487C24CBC93BED23C75C0934064448445A53D14BD14435E3E5441DEB91A4243B45544A746023AB0C195C1A639403E95C2D0C0043557C5CE412E458DC2A23C883A43C116BCA4BF1BC4FD3CF73AF23A42B4F0454FBF9526B24336C41E3DBABDA6BFC33A57BE383D20452A4733C863B4E5ACC2444CBC3E381DC0CF3B28C237C348C221C6ECBCC14230441835A74380444A3DA2394C40E63C00BE01B9BAC19AC4313DF6B9DA4603C830B6E6C4E8C5C3BB0C3658C7C342DD3F38410CC3B13FE7B942BC1D3F42BE13C5F1C0D5B0DD3C3A3C193C7A40EC456F3880C6"> : tensor<4x2x3x5xf16>
    return %0 : tensor<4x2x3x5xf16>
  }
}

