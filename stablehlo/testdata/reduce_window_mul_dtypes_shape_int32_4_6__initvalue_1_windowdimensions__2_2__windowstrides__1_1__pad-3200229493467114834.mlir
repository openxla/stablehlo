// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xi32>
    %1 = call @expected() : () -> tensor<3x5xi32>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xi32>, tensor<i32>) -> tensor<3x5xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xi32>, tensor<3x5xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xi32> {
    %0 = stablehlo.constant dense<[[-1, -1, 0, 0, 1, 2], [-2, -1, -1, 4, -3, -3], [0, 4, -1, 1, 2, 7], [1, 6, 6, 3, 4, -2]]> : tensor<4x6xi32>
    return %0 : tensor<4x6xi32>
  }
  func.func private @expected() -> tensor<3x5xi32> {
    %0 = stablehlo.constant dense<[[2, 0, 0, 0, 18], [0, -4, 4, -24, 126], [0, -144, -18, 24, -112]]> : tensor<3x5xi32>
    return %0 : tensor<3x5xi32>
  }
}

