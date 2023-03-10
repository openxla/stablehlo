// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xui8>, tensor<2x7xui8>)
    %2 = call @expected() : () -> tensor<5x6x7xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2xi32>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<2x7xui8>) {
    %0 = stablehlo.constant dense<"0x000001000102060100000201010400020202060005000300020003010300030001000002020001010501030604010102020002020000060002060301000002000000000301030302020200010001030402030003030300020300000701000304000105030302020001010206010206000202030202030201030501000104040303020202010200020502000003000004080001000005010101030002020200040101020102010000000201040203060001000005010401000202050003010101050203020303020005020201050102020001"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[0, 4, 0, 2, 1, 1, 4], [0, 1, 2, 1, 0, 0, 0]]> : tensor<2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x000001000102060000000201010400020202060005000300020003010300030001000002020001010501030604010102020002020000060002060301000002000000000301030302020200010001030402030003030300020300000701000304000105030302020001000102010000000202030202030201030501000104040303020202010200020502000003000004080001000005010101030002020200040101020102010000000201040203060001000005010401000202050003010101050203020303020005020201050102020001"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

