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
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xf16>, tensor<1xi32>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16>, tensor<1x3xf16>) {
    %0 = stablehlo.constant dense<"0x5CBC3A3D91375F3B38C4DD4330B99D3E3AB6BB413CC3933624BB5C42244530C033BCC73D09429E355E40B4BC84C14C41183CD0B4A73C4D46653DF34089BF1D4350BB6F39073DC83EF4BD224442BC1141A642F3C1D84385BF60C10C4035C1AE3AEFBDC83C67BDD6C1734695BEA83FFF3E3EC1A5C2D33E633AC0BDBE3A8BBB29C24A407D3DD0BBADC5D2C44540E445BBBBC1B7F1321E3DF23BD4C3B6BF4BBC2ABD80BE3ABCA2415EC476C0D33A6A3991431CC005C4BA43AE3DE9384B43E1C4E73952C2C6404D40F73D993C6B34B746C02F90C030B0F8C3963E99C1604365C697448DC034C274B999BF55396E40E3418B39D2C2793F3B42C44562BF63C61441A0453F3D1F3184404C41BAC109433840C44393B5C43C302CF3BA8B3CE1BE8C3732BE21C289B98D4426C44BC0C539"> : tensor<1x50x3xf16>
    %1 = stablehlo.constant dense<[[4.604490e-01, -1.554690e+00, -1.485350e+00]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x50x3xf16> {
    %0 = stablehlo.constant dense<"0x5CBC3A3D91375F3B38C4DD4330B99D3E3AB6BB413CC3933624BB5C42244530C033BCC73D09429E355E40B4BC84C14C41183CD0B4A73C4D46653DF34089BF1D4350BB6F39073DC83EF4BD224442BC1141A642F3C1D84385BF60C10C4035C1AE3AEFBDC83C67BDD6C1734695BEA83FFF3E3EC1A5C2D33E633AC0BDBE3A8BBB29C24A407D3DD0BBADC5D2C44540E445BBBBC1B7F1321E3DF23BD4C3B6BF4BBC2ABD80BE3ABCA2415EC476C0D33A6A3991431CC005C4BA43AE3DE9384B43E1C4E73966C1A83A5239F73D993C6B34B746C02F90C030B0F8C3963E99C1604365C697448DC034C274B999BF55396E40E3418B39D2C2793F3B42C44562BF63C61441A0453F3D1F3184404C41BAC109433840C44393B5C43C302CF3BA8B3CE1BE8C3732BE21C289B98D4426C44BC0C539"> : tensor<1x50x3xf16>
    return %0 : tensor<1x50x3xf16>
  }
}

