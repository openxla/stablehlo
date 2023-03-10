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
      stablehlo.return %arg1 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xui8>, tensor<1xi32>, tensor<1x3xui8>) -> tensor<1x50x3xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xui8>, tensor<1x50x3xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xui8>, tensor<1x3xui8>) {
    %0 = stablehlo.constant dense<"0x020004020200010400010105070002020402010201000105000503000900010301030100040004010001020007050505000402060000020204040405030001040102000100030103020102000201020101030201050103000100030407000002000101000702040003050502000100010301000500000201010005010103000003000001010000000304000300030202030003010301"> : tensor<1x50x3xui8>
    %1 = stablehlo.constant dense<[[1, 1, 2]]> : tensor<1x3xui8>
    return %0, %1 : tensor<1x50x3xui8>, tensor<1x3xui8>
  }
  func.func private @expected() -> tensor<1x50x3xui8> {
    %0 = stablehlo.constant dense<"0x020004020200010400010105070002020402010201000105000503000900010301030100040004010001020007050505000402060000020204040405030001040102000100030103020102000201020101030201050103000100030407000002010102000702040003050502000100010301000500000201010005010103000003000001010000000304000300030202030003010301"> : tensor<1x50x3xui8>
    return %0 : tensor<1x50x3xui8>
  }
}

