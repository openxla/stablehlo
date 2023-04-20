// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xi8>, tensor<1x3xi8>)
    %2 = call @expected() : () -> tensor<1x50x3xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi8>, tensor<1xi32>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi8>, tensor<1x3xi8>) {
    %0 = stablehlo.constant dense<"0xFE00FD000003FD04FF00FF00FF0200FF050000FB0100010300FFFE0200030105FF0000FE0400FF020100FE010402FFFA030000FF000200010002F9FE01FE010004FD06000004FF000000FEFE00FCFD0001FDFEF80100FC00FFFF00FC0002000204FE00FE00FC0800000502FE0005FEFE04030300FA01FA0002FC00FBFBFDFEFE02010400FDFE00020003FBFF0200F700FF0100FE00FD"> : tensor<1x50x3xi8>
    %1 = stablehlo.constant dense<[[-1, 6, -2]]> : tensor<1x3xi8>
    return %0, %1 : tensor<1x50x3xi8>, tensor<1x3xi8>
  }
  func.func private @expected() -> tensor<1x50x3xi8> {
    %0 = stablehlo.constant dense<"0xFE00FD000003FD04FF00FF00FF0200FF050000FB0100010300FFFE0200030105FF0000FE0400FF020100FE010402FFFA030000FF000200010002F9FE01FE010004FD06000004FF000000FEFE00FCFD0001FDFEF80100FC00FFFF00FC00020002FFFEFEFE00FC0800000502FE0005FEFE04030300FA01FA0002FC00FBFBFDFEFE02010400FDFE00020003FBFF0200F700FF0100FE00FD"> : tensor<1x50x3xi8>
    return %0 : tensor<1x50x3xi8>
  }
}

