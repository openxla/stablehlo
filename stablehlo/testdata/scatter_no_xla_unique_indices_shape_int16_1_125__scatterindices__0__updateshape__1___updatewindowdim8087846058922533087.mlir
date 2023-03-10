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
      stablehlo.return %arg1 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xi16>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x125xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xi16>, tensor<1x125xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xi16>, tensor<1xi16>) {
    %0 = stablehlo.constant dense<"0x0000FCFFFDFF0300FDFF0600FEFFFDFF00000000FFFF0000FCFFFEFFFFFF0300000000000400FCFF0100FFFFFFFFFFFF02000500FFFF0200FBFF0200FBFF00000100FFFF0000FFFF040005000000000000000200FBFF0200030003000200020004000100000000000100FEFF05000100060005000100FFFF01000000020003000400FEFF0000FFFF010000000000FBFF000000000200FFFFFFFF040000000300020001000400FDFFFBFF03000400FFFFFFFF0000FFFF0200FFFF05000300020000000500FEFF0000020000000200FEFFFFFF000001000300FEFFFFFFFCFF00000300FFFF010003000000FFFFFCFF01000500FBFFFBFF06000000"> : tensor<1x125xi16>
    %1 = stablehlo.constant dense<0> : tensor<1xi16>
    return %0, %1 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> tensor<1x125xi16> {
    %0 = stablehlo.constant dense<"0x0000FCFFFDFF0300FDFF0600FEFFFDFF00000000FFFF0000FCFFFEFFFFFF0300000000000400FCFF0100FFFFFFFFFFFF02000500FFFF0200FBFF0200FBFF00000100FFFF0000FFFF040005000000000000000200FBFF0200030003000200020004000100000000000100FEFF05000100060005000100FFFF01000000020003000400FEFF0000FFFF010000000000FBFF000000000200FFFFFFFF040000000300020001000400FDFFFBFF03000400FFFFFFFF0000FFFF0200FFFF05000300020000000500FEFF0000020000000200FEFFFFFF000001000300FEFFFFFFFCFF00000300FFFF010003000000FFFFFCFF01000500FBFFFBFF06000000"> : tensor<1x125xi16>
    return %0 : tensor<1x125xi16>
  }
}

