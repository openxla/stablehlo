// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xui8>
    %1 = call @expected() : () -> tensor<3x5xui8>
    %2 = stablehlo.constant dense<255> : tensor<ui8>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui8>) -> tensor<ui8>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %6 = stablehlo.minimum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %6 : tensor<ui8>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xui8>, tensor<ui8>) -> tensor<3x5xui8>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xui8>, tensor<3x5xui8>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xui8> {
    %0 = stablehlo.constant dense<[[3, 2, 0, 5, 0, 3], [3, 1, 1, 3, 4, 4], [3, 1, 1, 1, 2, 1], [7, 1, 0, 1, 2, 2]]> : tensor<4x6xui8>
    return %0 : tensor<4x6xui8>
  }
  func.func private @expected() -> tensor<3x5xui8> {
    %0 = stablehlo.constant dense<[[1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 1, 1]]> : tensor<3x5xui8>
    return %0 : tensor<3x5xui8>
  }
}

