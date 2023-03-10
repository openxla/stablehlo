// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xbf16>, tensor<1xbf16>)
    %2 = call @expected() : () -> tensor<1x125xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xbf16>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1x125xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xbf16>, tensor<1xbf16>) {
    %0 = stablehlo.constant dense<"0x2ABEBF3F03405EC08BBFA33F383F11C0DF40593E1BC08A40DEBE4F3ED7BEC7C02540F5BD8A3FA73F33408640733E8E3F24C0DABF1DC0844014403E40C33FE2BDA340CF40A7BF3CBEFDC099C069BF87BFB03E4ABC8EC007BF52C0B7BFD7BFE040AE40343FECBE023DCBC08FBFAE3E15C065C0B2C082BF2EBF82402F40E4BF8AC0A6C035C0ADBF1F3E6F3F364017C0064001C08E3E8B4094BF393F8CC04A408ABF323F8EC01A408E3D90C000C0B83FC73EA940C540DABF9C40D93F214001401A3FE640C03FACC04140D13E134094BE3B409EBEA2BEB93E904082C0A0408D407E3FCBBF0ABF1BBF1DBF7AC04040C1BF743F1BC019BFED3F2540E6BF"> : tensor<1x125xbf16>
    %1 = stablehlo.constant dense<-2.640630e+00> : tensor<1xbf16>
    return %0, %1 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> tensor<1x125xbf16> {
    %0 = stablehlo.constant dense<"0xE03EBF3F03405EC08BBFA33F383F11C0DF40593E1BC08A40DEBE4F3ED7BEC7C02540F5BD8A3FA73F33408640733E8E3F24C0DABF1DC0844014403E40C33FE2BDA340CF40A7BF3CBEFDC099C069BF87BFB03E4ABC8EC007BF52C0B7BFD7BFE040AE40343FECBE023DCBC08FBFAE3E15C065C0B2C082BF2EBF82402F40E4BF8AC0A6C035C0ADBF1F3E6F3F364017C0064001C08E3E8B4094BF393F8CC04A408ABF323F8EC01A408E3D90C000C0B83FC73EA940C540DABF9C40D93F214001401A3FE640C03FACC04140D13E134094BE3B409EBEA2BEB93E904082C0A0408D407E3FCBBF0ABF1BBF1DBF7AC04040C1BF743F1BC019BFED3F2540E6BF"> : tensor<1x125xbf16>
    return %0 : tensor<1x125xbf16>
  }
}

