// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xf16>, tensor<1xf16>)
    %2 = call @expected() : () -> tensor<1x125xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xf16>, tensor<1xi32>, tensor<1xf16>) -> tensor<1x125xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xf16>, tensor<1x125xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xf16>, tensor<1xf16>) {
    %0 = stablehlo.constant dense<"0xDB36B04410BA0BC33F4158BC1F38BE393D41EDC514BCF6BCDAB421BEC8C5634103395AC2DC4003463640D840C4BEC73DA641473C95C406C12EC26F4386C45FC4494031382C45AA388EC0B8BAC045E33FB4B57C431D3E0E3E92C4E8BD3DC4AD412839DAC2C9C3BA3FB4B6EAB6D1C3BD3FDCC0A1C47F34A4BF66BF2146C74421C2C7408EB48EC7623A3FBFEF3CD033F5B925B492363E40E9BE5A4376427642F6B5B04137BE34C07235B9BF203E83C7873EDFC2F8B55A4402B7C1C1EEC4353E84C09DC4D0C769BCD1BD313EA0B99BBD76BEEEC2C7C29DC4FD3EA4C3DAB9DD3F86AD30C0E0BEED3674C02D4482C26CB621BC70BD95C31E43AD46EBBD"> : tensor<1x125xf16>
    %1 = stablehlo.constant dense<1.951170e+00> : tensor<1xf16>
    return %0, %1 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> tensor<1x125xf16> {
    %0 = stablehlo.constant dense<"0xDB36B04410BA0BC33F4158BC1F38BE393D41EDC514BCF6BCDAB421BEC8C5634103395AC2DC4003463640D840C4BEC73DA641473C95C406C12EC26F4386C45FC4494031382C45AA388EC0B8BAC045E33FB4B57C431D3E0E3E92C4E8BD3DC4AD412839DAC2C9C3BA3FB4B6EAB6D1C3BD3FDCC0A1C47F34A4BF66BF2146C74421C2C7408EB48EC7623A3FBFEF3CD033F5B925B492363E40E9BE5A4376427642F6B5B04137BE34C07235B9BF203E83C7873EDFC2F8B55A4402B7C1C1EEC4353E84C09DC4D0C769BCD1BD313EA0B99BBD76BEEEC2C7C29DC4FD3EA4C3DAB9DD3F86AD30C0E0BEED3674C02D4482C26CB621BC70BD95C31E43AD46EBBD"> : tensor<1x125xf16>
    return %0 : tensor<1x125xf16>
  }
}

