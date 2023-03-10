// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xi16>, tensor<1xi16>)
    %2 = call @expected() : () -> tensor<1x125xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xi16>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x125xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xi16>, tensor<1x125xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xi16>, tensor<1xi16>) {
    %0 = stablehlo.constant dense<"0x02000300010004000000FFFF0100FEFFFDFF0100FDFF0000FFFFFDFF01000200FDFFFDFFFEFFFDFF0100FEFF05000200F9FFFFFFFFFF0200FCFFFEFF010000000A000000030001000300FDFFFEFF0000FDFF0000FCFF02000100FBFFFFFF0500FFFF0400000000000400FEFFFFFF000002000200FDFF0300FEFF0300FFFF050001000100FCFF0300000001000000020000000000040001000000000003000000FFFF0300040000000000000000000400040001000100FEFF0100010000000000F7FF0700FFFF0200FCFF03000000FFFF03000100000001000100000001000100FDFFFFFFFFFFF9FF040000000300FCFFFDFF0200010000000200"> : tensor<1x125xi16>
    %1 = stablehlo.constant dense<-3> : tensor<1xi16>
    return %0, %1 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> tensor<1x125xi16> {
    %0 = stablehlo.constant dense<"0xFDFF0300010004000000FFFF0100FEFFFDFF0100FDFF0000FFFFFDFF01000200FDFFFDFFFEFFFDFF0100FEFF05000200F9FFFFFFFFFF0200FCFFFEFF010000000A000000030001000300FDFFFEFF0000FDFF0000FCFF02000100FBFFFFFF0500FFFF0400000000000400FEFFFFFF000002000200FDFF0300FEFF0300FFFF050001000100FCFF0300000001000000020000000000040001000000000003000000FFFF0300040000000000000000000400040001000100FEFF0100010000000000F7FF0700FFFF0200FCFF03000000FFFF03000100000001000100000001000100FDFFFFFFFFFFF9FF040000000300FCFFFDFF0200010000000200"> : tensor<1x125xi16>
    return %0 : tensor<1x125xi16>
  }
}

