// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xi16>, tensor<1x3xi16>)
    %2 = call @expected() : () -> tensor<1x50x3xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi16>, tensor<1xi32>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi16>, tensor<1x3xi16>) {
    %0 = stablehlo.constant dense<"0x03000300FFFF00000000FEFFFDFF0000FCFF00000300000000000000FEFF010001000000FFFF0000FEFF0100F9FF02000000FCFFFFFF020000000200FCFF000006000200FEFF03000000030000000000010002000000FEFF010002000000000001000600FEFFFDFF000003000000FFFF03000500000001000200FFFF030004000000FCFF000000000500FFFF00000100FDFFFFFF000002000100FFFF000003000100FFFF0200FDFFFEFFFFFF0100FFFF0000FEFFFFFF000003000000FDFFFEFFFFFFFEFF0300FDFF010001000400FEFF000002000000FBFFFFFFFFFFFBFF0400FFFFFCFF00000300FAFFFDFF0000FEFFFCFF0000FFFFFFFFFCFF020000000300FFFFF9FF02000100FEFFFEFFFFFFFAFF0000FEFF00000000000003000300FDFFFFFF0100000003000000F8FF"> : tensor<1x50x3xi16>
    %1 = stablehlo.constant dense<[[3, 1, 2]]> : tensor<1x3xi16>
    return %0, %1 : tensor<1x50x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> tensor<1x50x3xi16> {
    %0 = stablehlo.constant dense<"0x03000300FFFF00000000FEFFFDFF0000FCFF00000300000000000000FEFF010001000000FFFF0000FEFF0100F9FF02000000FCFFFFFF020000000200FCFF000006000200FEFF03000000030000000000010002000000FEFF010002000000000001000600FEFFFDFF000003000000FFFF03000500000001000200FFFF030004000000FCFF000000000500FFFF00000100FDFFFFFF000002000100FFFF000003000100FFFF0200FDFFFEFFFFFF0100FFFF0000FEFFFFFF000003000000FDFFFEFFFDFFFEFF0600FDFF010001000400FEFF000002000000FBFFFFFFFFFFFBFF0400FFFFFCFF00000300FAFFFDFF0000FEFFFCFF0000FFFFFFFFFCFF020000000300FFFFF9FF02000100FEFFFEFFFFFFFAFF0000FEFF00000000000003000300FDFFFFFF0100000003000000F8FF"> : tensor<1x50x3xi16>
    return %0 : tensor<1x50x3xi16>
  }
}

