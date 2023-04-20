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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xf16>, tensor<1xi32>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16>, tensor<1x3xf16>) {
    %0 = stablehlo.constant dense<"0xDD40B7C565C8AB46F3B7C3BDEA342F40D443C8C603B7F340149E9EC156AA83C4C3BD19BEF0BF4FBC024073C204C1904490402CC0BB44F4BE73C21FBEECC22E4438C279BAC0B511B82D41E936E03E4C362A2DA9C261362FC4DCB8D03EBCC6B1B967BBA6C74DC429391242BEBF6BC4D0C3423D5BB9EB3C5D4100C1E3BD5D3F673C8EBFF4BEDE32A3BFCABC57BEC142F6BA59B4DCBC1540FFBE8C44344095C48BC0B6BE67BC8640A5432EC1FAB894C071C4E1BFB63C4ABD0E3B5739AB42B2BE1F44AC3FA43C3841F534A3C3D84193B91FBD2540BFC41E3C38BE0539A9421C2D59BE4AC1ACAFB13B07413DB5BF42A947223F7744DBB9CB3CC9BE1FC46144233E19C6863FDFBE07418CC1AB3AA4C4A5C1ABC5D8382EC525C397C01242BF40FA409CBD2BC0BEB80E3EB0C20E2F79BC"> : tensor<1x50x3xf16>
    %1 = stablehlo.constant dense<[[4.660640e-01, -2.798830e+00, -1.446290e+00]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x50x3xf16> {
    %0 = stablehlo.constant dense<"0xDD40B7C565C8AB46F3B7C3BDEA342F40D443C8C603B7F340149E9EC156AA83C4C3BD19BEF0BF4FBC024073C204C1904490402CC0BB44F4BE73C21FBEECC22E4438C279BAC0B511B82D41E936E03E4C362A2DA9C261362FC4DCB8D03EBCC6B1B967BBA6C74DC429391242BEBF6BC4D0C3423D5BB9EB3C5D4100C1E3BD5D3F673C8EBFF4BEDE32A3BFCABC57BEC142F6BA59B4DCBC1540FFBE8C44344095C48BC0B6BE67BC8640A5432EC1FAB894C071C4E1BFB63C4ABD0E3B5739AB42B2BE1F44753799C1C9BDF534A3C3D84193B91FBD2540BFC41E3C38BE0539A9421C2D59BE4AC1ACAFB13B07413DB5BF42A947223F7744DBB9CB3CC9BE1FC46144233E19C6863FDFBE07418CC1AB3AA4C4A5C1ABC5D8382EC525C397C01242BF40FA409CBD2BC0BEB80E3EB0C20E2F79BC"> : tensor<1x50x3xf16>
    return %0 : tensor<1x50x3xf16>
  }
}

