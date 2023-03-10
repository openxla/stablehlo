// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>)
    %2 = call @expected() : () -> tensor<4x2x3x5xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi8>, tensor<2xi32>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>) {
    %0 = stablehlo.constant dense<"0x02FB02FFFC08FEFCFF01FF000000FE0003FD030003FDFCFE0002010202FEFD020000FD00FC04FF00FD00FB0100030102FF06000103010000FD0003F8020503FF02FCFCFEFF01FCFEFFFD0301FF0407FE03FE0000FEFDFBFD0200FE0401FFFE03FEFF0000FF02FE030000FB0005010400FE0200FD0001FD00"> : tensor<4x2x3x5xi8>
    %1 = stablehlo.constant dense<[[-2, 4, 5], [-2, 4, -6], [2, 0, 4], [-6, 4, 1]]> : tensor<4x3xi8>
    return %0, %1 : tensor<4x2x3x5xi8>, tensor<4x3xi8>
  }
  func.func private @expected() -> tensor<4x2x3x5xi8> {
    %0 = stablehlo.constant dense<"0x02FB02FFFA08FEFCFF05FF000000030003FD030003FDFCFE0002010202FEFD020000FB00FC04FF04FD00FB01FA030102FF06000103010000FD0003F8020503FF04FCFCFEFF01FCFEFFFD0701FF0407FE03FE0000FEFDFBFD0200FE0401FFF803FEFF0004FF02FE030100FB0005010400FE0200FD0001FD00"> : tensor<4x2x3x5xi8>
    return %0 : tensor<4x2x3x5xi8>
  }
}

