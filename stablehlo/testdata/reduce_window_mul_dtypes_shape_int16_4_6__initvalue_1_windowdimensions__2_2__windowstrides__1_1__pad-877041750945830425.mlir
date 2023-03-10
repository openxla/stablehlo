// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xi16>
    %1 = call @expected() : () -> tensor<3x5xi16>
    %2 = stablehlo.constant dense<1> : tensor<i16>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xi16>, tensor<i16>) -> tensor<3x5xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xi16>, tensor<3x5xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xi16> {
    %0 = stablehlo.constant dense<[[3, 2, 0, 2, -3, 0], [-3, 3, 2, -1, 2, 0], [-2, -1, 0, 7, -3, -3], [1, -2, 2, 4, 1, 1]]> : tensor<4x6xi16>
    return %0 : tensor<4x6xi16>
  }
  func.func private @expected() -> tensor<3x5xi16> {
    %0 = stablehlo.constant dense<[[-54, 0, 0, 12, 0], [-18, 0, 0, 42, 0], [-4, 0, 0, -84, 9]]> : tensor<3x5xi16>
    return %0 : tensor<3x5xi16>
  }
}

