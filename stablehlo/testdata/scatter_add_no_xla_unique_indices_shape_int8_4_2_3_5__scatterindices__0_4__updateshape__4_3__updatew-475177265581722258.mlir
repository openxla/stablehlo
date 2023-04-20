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
    %0 = stablehlo.constant dense<"0xFE03FB0000FE05FCFB06030000FFFDF80305FE00FD02040100FD0201030400FFFF02FF0300000001FC01FEFEFDFEFEFF030002FF03010003FFFE03FC02FC02020000000200FD0101FF0000FD01FE02FDFE0000030003FDFDFCFC0304000101FEFFFFFCFD0304030001FF020007FAF9FD00FF00030400FCFF"> : tensor<4x2x3x5xi8>
    %1 = stablehlo.constant dense<[[-1, -1, -3], [-5, 0, -1], [1, 2, -2], [0, 3, 2]]> : tensor<4x3xi8>
    return %0, %1 : tensor<4x2x3x5xi8>, tensor<4x3xi8>
  }
  func.func private @expected() -> tensor<4x2x3x5xi8> {
    %0 = stablehlo.constant dense<"0xFE03FB00FFFE05FCFB05030000FFFAF80305FE00FD02040100FD0201030400FFFF02FA0300000001FC01FEFEFCFEFEFF030002FF03010003FFFE03FC02FC02020100000200FF0101FF00FEFD01FE02FDFE0000030003FDFDFCFC0304000101FEFFFFFC000304030003FF020007FAF9FD00FF00030400FCFF"> : tensor<4x2x3x5xi8>
    return %0 : tensor<4x2x3x5xi8>
  }
}

