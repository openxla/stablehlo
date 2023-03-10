// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xi8>
    %1 = call @expected() : () -> tensor<3x5xi8>
    %2 = stablehlo.constant dense<1> : tensor<i8>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xi8>, tensor<i8>) -> tensor<3x5xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xi8>, tensor<3x5xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xi8> {
    %0 = stablehlo.constant dense<[[0, 0, 1, 1, -2, 1], [0, -6, -1, -4, -3, -5], [2, -1, 4, 0, 0, -4], [0, -1, 0, -1, 0, 0]]> : tensor<4x6xi8>
    return %0 : tensor<4x6xi8>
  }
  func.func private @expected() -> tensor<3x5xi8> {
    %0 = stablehlo.constant dense<[[1, 1, 1, 1, 1], [2, 4, 4, 1, 1], [2, 4, 4, 1, 1]]> : tensor<3x5xi8>
    return %0 : tensor<3x5xi8>
  }
}

