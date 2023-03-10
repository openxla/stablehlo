// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xui8>, tensor<1xui8>)
    %2 = call @expected() : () -> tensor<1x125xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      stablehlo.return %arg1 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xui8>, tensor<1xi32>, tensor<1xui8>) -> tensor<1x125xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xui8>, tensor<1x125xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xui8>, tensor<1xui8>) {
    %0 = stablehlo.constant dense<"0x0601050404050404020000030003010402000306030100010302010201040102020101010003020101070201020100010001020405020201000104000000030101000101050000070204000202050000060201030501010501050105010101030001030101000100020101060401040304030303000201000402020003"> : tensor<1x125xui8>
    %1 = stablehlo.constant dense<0> : tensor<1xui8>
    return %0, %1 : tensor<1x125xui8>, tensor<1xui8>
  }
  func.func private @expected() -> tensor<1x125xui8> {
    %0 = stablehlo.constant dense<"0x0001050404050404020000030003010402000306030100010302010201040102020101010003020101070201020100010001020405020201000104000000030101000101050000070204000202050000060201030501010501050105010101030001030101000100020101060401040304030303000201000402020003"> : tensor<1x125xui8>
    return %0 : tensor<1x125xui8>
  }
}

