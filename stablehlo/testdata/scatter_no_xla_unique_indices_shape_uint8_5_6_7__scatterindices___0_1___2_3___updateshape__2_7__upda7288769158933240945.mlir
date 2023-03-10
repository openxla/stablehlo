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
      stablehlo.return %arg1 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2xi32>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<2x7xui8>) {
    %0 = stablehlo.constant dense<"0x000001010202030401000305000301020100000307010204040204040002000404000001050101040400000501010204050504000406070000030302010100040203000303000600050209000300010201010402030000020100020303010001020101010305020303000008020200040400010200000203040202030102010000010201010303010209020301030306000600000404020200000502000104020101040501000102000004020101000002030101000704040100040105050203040303010303030302090200020101010200"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[1, 3, 2, 0, 0, 1, 5], [0, 0, 4, 0, 3, 2, 1]]> : tensor<2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x000001010202030103020000010501020100000307010204040204040002000404000001050101040400000501010204050504000406070000030302010100040203000303000600050209000300010201010402030000020100020303010001020101010305020303000004000302010400010200000203040202030102010000010201010303010209020301030306000600000404020200000502000104020101040501000102000004020101000002030101000704040100040105050203040303010303030302090200020101010200"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

