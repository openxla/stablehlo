// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xi16>
    %1 = call @expected() : () -> tensor<3x5xi16>
    %2 = stablehlo.constant dense<-32768> : tensor<i16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i16>) -> tensor<i16>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %6 = stablehlo.maximum %arg0, %arg1 : tensor<i16>
      stablehlo.return %6 : tensor<i16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xi16>, tensor<i16>) -> tensor<3x5xi16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xi16>, tensor<3x5xi16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xi16> {
    %0 = stablehlo.constant dense<[[0, -5, 0, 3, 1, 0], [3, -1, -6, -2, 2, 0], [0, 2, 2, -2, -5, -2], [-8, -4, 0, 1, 2, -3]]> : tensor<4x6xi16>
    return %0 : tensor<4x6xi16>
  }
  func.func private @expected() -> tensor<3x5xi16> {
    %0 = stablehlo.constant dense<[[3, 0, 3, 3, 2], [3, 2, 2, 2, 2], [2, 2, 2, 2, 2]]> : tensor<3x5xi16>
    return %0 : tensor<3x5xi16>
  }
}

