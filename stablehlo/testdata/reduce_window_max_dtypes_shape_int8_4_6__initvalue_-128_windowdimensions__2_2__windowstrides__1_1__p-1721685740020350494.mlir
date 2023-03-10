// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xi8>
    %1 = call @expected() : () -> tensor<3x5xi8>
    %2 = stablehlo.constant dense<-128> : tensor<i8>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i8>) -> tensor<i8>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %6 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %6 : tensor<i8>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xi8>, tensor<i8>) -> tensor<3x5xi8>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xi8>, tensor<3x5xi8>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xi8> {
    %0 = stablehlo.constant dense<[[0, 0, 0, -1, -2, -3], [-1, 2, -2, 1, 0, -6], [0, 8, -4, 4, -1, 2], [-1, 1, 5, 0, -1, 1]]> : tensor<4x6xi8>
    return %0 : tensor<4x6xi8>
  }
  func.func private @expected() -> tensor<3x5xi8> {
    %0 = stablehlo.constant dense<[[2, 2, 1, 1, 0], [8, 8, 4, 4, 2], [8, 8, 5, 4, 2]]> : tensor<3x5xi8>
    return %0 : tensor<3x5xi8>
  }
}

