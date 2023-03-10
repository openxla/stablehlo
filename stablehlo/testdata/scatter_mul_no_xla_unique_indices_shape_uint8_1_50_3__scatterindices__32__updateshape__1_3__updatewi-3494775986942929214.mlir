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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xui8>, tensor<1xi32>, tensor<1x3xui8>) -> tensor<1x50x3xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xui8>, tensor<1x50x3xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xui8>, tensor<1x3xui8>) {
    %0 = stablehlo.constant dense<"0x010101030001010303000601010000020203010102020102050505020001010200010603030304000200010103030501000204000103010007010301040101030000010100020101000403030002010202030602020200030401030100030003020204040308000702000001020501040001000103010100010107000100000300020102010001000301010003080102000200000103"> : tensor<1x50x3xui8>
    %1 = stablehlo.constant dense<[[3, 0, 2]]> : tensor<1x3xui8>
    return %0, %1 : tensor<1x50x3xui8>, tensor<1x3xui8>
  }
  func.func private @expected() -> tensor<1x50x3xui8> {
    %0 = stablehlo.constant dense<"0x010101030001010303000601010000020203010102020102050505020001010200010603030304000200010103030501000204000103010007010301040101030000010100020101000403030002010202030602020200030401030100030003060008040308000702000001020501040001000103010100010107000100000300020102010001000301010003080102000200000103"> : tensor<1x50x3xui8>
    return %0 : tensor<1x50x3xui8>
  }
}

