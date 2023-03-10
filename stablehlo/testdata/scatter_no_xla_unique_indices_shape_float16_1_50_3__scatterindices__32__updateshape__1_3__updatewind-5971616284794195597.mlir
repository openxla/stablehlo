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
    %0 = stablehlo.constant dense<"0x66C487C6DDC89A34E6B58EBD66C0DEBB6BB51BC7D5C45943FD3F7DBA54C1A93C0A3D652C04C46AC23BB7233CFAC16E454AC074323DC05944C9416347D6B988B78345B93653373EC3E4BD8A4113351DC027C09145593D1DA48D426CC64B3DCEC4FEBB96C26D369B37DBC02BC270C754C32E414EBEEF42D63C61C2CCBB643E26BE9CC86EC6D13492AE73393BAFFCBC743F62B8BE3E9DC17C36AFC373C436C188BB5C43CF3E8DBD6C46762D3141FA31B539A84092C172C28AC06E43F8BEECBF27BCFE41923F4EC3F43BC4BD3137993327B9AFC4A0B456C1A2472B43F8BD1432A4C113C181C20DBD5AC094BD2FB60B44433B1C39E7B96D40EFC3B128FD3BEF3C64C3F0A93E3C6E4067C4924218C448BCC2C215C159C85BC2933F8EC5874751BEDCBAF13FF635B1352043D34536C0"> : tensor<1x50x3xf16>
    %1 = stablehlo.constant dense<[[-1.777340e+00, 1.616210e+00, -2.300780e+00]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x50x3xf16> {
    %0 = stablehlo.constant dense<"0x66C487C6DDC89A34E6B58EBD66C0DEBB6BB51BC7D5C45943FD3F7DBA54C1A93C0A3D652C04C46AC23BB7233CFAC16E454AC074323DC05944C9416347D6B988B78345B93653373EC3E4BD8A4113351DC027C09145593D1DA48D426CC64B3DCEC4FEBB96C26D369B37DBC02BC270C754C32E414EBEEF42D63C61C2CCBB643E26BE9CC86EC6D13492AE73393BAFFCBC743F62B8BE3E9DC17C36AFC373C436C188BB5C43CF3E8DBD6C46762D3141FA31B539A84092C172C28AC06E43F8BEECBF27BC1CBF773E9AC0F43BC4BD3137993327B9AFC4A0B456C1A2472B43F8BD1432A4C113C181C20DBD5AC094BD2FB60B44433B1C39E7B96D40EFC3B128FD3BEF3C64C3F0A93E3C6E4067C4924218C448BCC2C215C159C85BC2933F8EC5874751BEDCBAF13FF635B1352043D34536C0"> : tensor<1x50x3xf16>
    return %0 : tensor<1x50x3xf16>
  }
}

