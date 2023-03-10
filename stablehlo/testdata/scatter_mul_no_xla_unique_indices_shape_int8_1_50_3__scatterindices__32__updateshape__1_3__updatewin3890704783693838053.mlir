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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi8>, tensor<1xi32>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi8>, tensor<1x3xi8>) {
    %0 = stablehlo.constant dense<"0x04FF00FB0102FD050200FD000100FD0106FE0101FFFC01FD00FF02000202FFFF0201FE0100010005FCFF00000001010100FE0001FCFE000203FD01FE0003010100030108FE02030002FFFA00FFFFFB0400FFFBFEFF01FE010200FF0001FE0103040001010002FC0103FE010001010BFF000502FD0100FDFD01050101FC0002FE00FF01030400000203FF0005FD000302FB0003010200"> : tensor<1x50x3xi8>
    %1 = stablehlo.constant dense<[[3, -4, 4]]> : tensor<1x3xi8>
    return %0, %1 : tensor<1x50x3xi8>, tensor<1x3xi8>
  }
  func.func private @expected() -> tensor<1x50x3xi8> {
    %0 = stablehlo.constant dense<"0x04FF00FB0102FD050200FD000100FD0106FE0101FFFC01FD00FF02000202FFFF0201FE0100010005FCFF00000001010100FE0001FCFE000203FD01FE0003010100030108FE02030002FFFA00FFFFFB0400FFFBFEFF01FE010200FF0001FE01030C0004010002FC0103FE010001010BFF000502FD0100FDFD01050101FC0002FE00FF01030400000203FF0005FD000302FB0003010200"> : tensor<1x50x3xi8>
    return %0 : tensor<1x50x3xi8>
  }
}

