// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xui8>, tensor<1x3xui8>)
    %2 = call @expected() : () -> tensor<1x50x3xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xui8>, tensor<1xi32>, tensor<1x3xui8>) -> tensor<1x50x3xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xui8>, tensor<1x50x3xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xui8>, tensor<1x3xui8>) {
    %0 = stablehlo.constant dense<"0x020405050107000200030001000301010302010200010102040005030005020000020303000503050304010002040101010101000200050605060000010100010405030201030203010001010303010002010500060002000603000500000107000002010600020103010600000102010000040100000105010001000101020300020202020002030403010004000001000101020003"> : tensor<1x50x3xui8>
    %1 = stablehlo.constant dense<[[2, 4, 1]]> : tensor<1x3xui8>
    return %0, %1 : tensor<1x50x3xui8>, tensor<1x3xui8>
  }
  func.func private @expected() -> tensor<1x50x3xui8> {
    %0 = stablehlo.constant dense<"0x020405050107000200030001000301010302010200010102040005030005020000020303000503050304010002040101010101000200050605060000010100010405030201030203010001010303010002010500060002000603000500000107020403010600020103010600000102010000040100000105010001000101020300020202020002030403010004000001000101020003"> : tensor<1x50x3xui8>
    return %0 : tensor<1x50x3xui8>
  }
}

