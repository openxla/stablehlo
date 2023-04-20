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
      stablehlo.return %arg1 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xf16>, tensor<1xi32>, tensor<1xf16>) -> tensor<1x125xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xf16>, tensor<1x125xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xf16>, tensor<1xf16>) {
    %0 = stablehlo.constant dense<"0x1243424560C442BC50C106BA0AC0C2B696415A3977AF47452F43D94258C15D36794295C0354105C17FC1B33BBE43C6C00145A64131C29EBD58BB30C16CC0F6C1DEBF08444C3D573FB5BC54C495AF01BE38C25B3B7C453538744109383E3D01407646B641844580B91C43A03C0841F33BE3C18FA8D239C2C15B3F6B41A23FBCC6B3B840C16FC26A3A663446B75140E0BE1E39A6C19BAD86B65E438B40C5410A24D6C1BC44B0C4E03B7EC1D4BD57BA66411748743C6B3D4242D24012C3BBB3FCC4823E46BC4F4011BB86C41CAC46B353447CC44EBC75C19EB0E1478FB575335841A2391742523684BB9839A943FDC53B46124435C6BBC2E3BD97BF"> : tensor<1x125xf16>
    %1 = stablehlo.constant dense<1.528320e+00> : tensor<1xf16>
    return %0, %1 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> tensor<1x125xf16> {
    %0 = stablehlo.constant dense<"0x1D3E424560C442BC50C106BA0AC0C2B696415A3977AF47452F43D94258C15D36794295C0354105C17FC1B33BBE43C6C00145A64131C29EBD58BB30C16CC0F6C1DEBF08444C3D573FB5BC54C495AF01BE38C25B3B7C453538744109383E3D01407646B641844580B91C43A03C0841F33BE3C18FA8D239C2C15B3F6B41A23FBCC6B3B840C16FC26A3A663446B75140E0BE1E39A6C19BAD86B65E438B40C5410A24D6C1BC44B0C4E03B7EC1D4BD57BA66411748743C6B3D4242D24012C3BBB3FCC4823E46BC4F4011BB86C41CAC46B353447CC44EBC75C19EB0E1478FB575335841A2391742523684BB9839A943FDC53B46124435C6BBC2E3BD97BF"> : tensor<1x125xf16>
    return %0 : tensor<1x125xf16>
  }
}

