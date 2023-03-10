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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xf16>, tensor<2xi32>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>) {
    %0 = stablehlo.constant dense<"0x0FC02040D23CEDB02A40CDBD333F3941183574C0D039B3BD8EC4A440034459C332466E40E63F0F386EC00A3110400B3E15C3FC41C0BAADC13445F3BB7D44DCC27D39E12B2CBC0EBCB4C151C14BB16C3CC6BEAC445DC072C4A9C3EC408EBB444063382CC138C28A430744E43CBB35C53FABBCD0BD823ABD31F04273B9AD478F44E6C2663D0BB773363EA98BB4BFC4E0419F3EE73A91C386C53ABD52BD59458542E8440C36D042F9457AC38AB60A3E49346D41403E8343B7C03C39473E4841CD45ED434E4128C2F540D0BBE144304196BD84447C414542ACC0693FAFB5C538C8ADAA3F733D214729444139CB44003E7DBB"> : tensor<4x2x3x5xf16>
    %1 = stablehlo.constant dense<[[-1.967770e+00, -7.055660e-01, -3.787110e+00], [2.255860e+00, 1.802730e+00, -3.796880e+00], [-3.437500e+00, -3.253910e+00, -5.937500e+00], [-1.982420e+00, -4.503910e+00, 3.605470e+00]]> : tensor<4x3xf16>
    return %0, %1 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xf16> {
    %0 = stablehlo.constant dense<"0x0FC02040D23CEDB0DFBFCDBD333F3941183574C0D039B3BD8EC4A44093C359C332466E40E63F0F386EC00A3110400B3E15C3FC41C0BAADC13445F3BB7D44DCC27D39E12B2CBC0EBCB4C151C14BB16C3CC6BEAC445DC072C4A9C3EC408EBB444063382CC138C28A430744E43CBB35C53FABBCD0BD823ABD31F04273B9AD478F44E6C2663D0BB773363EA982C2BFC4E0419F3EE73AF0C586C53ABD52BD59458542E8440C36D042F9457AC38AB60A3E49346D41403E8343B7C03C39473EEEBFCD45ED434E4128C281C4D0BBE144304196BD36437C414542ACC0693FAFB5C538C8ADAA3F733D214729444139CB44003E7DBB"> : tensor<4x2x3x5xf16>
    return %0 : tensor<4x2x3x5xf16>
  }
}

