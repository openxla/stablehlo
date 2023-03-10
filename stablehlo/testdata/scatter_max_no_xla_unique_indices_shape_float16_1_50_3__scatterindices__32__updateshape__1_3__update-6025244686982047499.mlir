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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xf16>, tensor<1xi32>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16>, tensor<1x3xf16>) {
    %0 = stablehlo.constant dense<"0x1ABC10BF5B3C9037813C9240543DBFB89BC256C0E18E313B8DBD813F88C28F40C9C118C1C941C5BC46C4B3BE3C3FC5BD48444EC5B042E6B72CC269BB334201C13540A2418536F3BEEB403C389B3E7E3911C381451EC4464045BEF23CF3C46B402AC225C7B0C0F94374C457397E3984BDCDBE7BC194409BBE17C089B0ACC545C1B4C588C359C27546AEC3F83E8BC1AC3AE8B1624340BCCE43653F0943D4BA61C046BE3045ACC3CA4111425541474560C544C06CC402C05DC404BFE6C029BB74C1364091C063B400C4DBC315404EC132C331C4E7C0E5C2BFB5BA3C243F3A42CEB04842643C364307C042BCB6318F3B2345262E47C2163B4E3A4A45CBBBC84669BCF8BE463D3844743C4E38A73C52C3B0435D40A33DCB31F0BE7ABA9644B2BC3F3809C40243A1C0F44575B8903D"> : tensor<1x50x3xf16>
    %1 = stablehlo.constant dense<[[6.801760e-01, -9.677730e-01, 6.070310e+00]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x50x3xf16> {
    %0 = stablehlo.constant dense<"0x1ABC10BF5B3C9037813C9240543DBFB89BC256C0E18E313B8DBD813F88C28F40C9C118C1C941C5BC46C4B3BE3C3FC5BD48444EC5B042E6B72CC269BB334201C13540A2418536F3BEEB403C389B3E7E3911C381451EC4464045BEF23CF3C46B402AC225C7B0C0F94374C457397E3984BDCDBE7BC194409BBE17C089B0ACC545C1B4C588C359C27546AEC3F83E8BC1AC3AE8B1624340BCCE43653F0943D4BA61C046BE3045ACC3CA4111425541474560C544C06CC402C05DC404BFE6C029BB74C13640BEBB124600C4DBC315404EC132C331C4E7C0E5C2BFB5BA3C243F3A42CEB04842643C364307C042BCB6318F3B2345262E47C2163B4E3A4A45CBBBC84669BCF8BE463D3844743C4E38A73C52C3B0435D40A33DCB31F0BE7ABA9644B2BC3F3809C40243A1C0F44575B8903D"> : tensor<1x50x3xf16>
    return %0 : tensor<1x50x3xf16>
  }
}

