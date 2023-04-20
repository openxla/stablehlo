// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x1x6xui16>, tensor<2x4x6xui16>)
    %1 = call @expected() : () -> tensor<2x4x6xui16>
    %2 = stablehlo.constant dense<0> : tensor<ui16>
    %3 = stablehlo.pad %0#1, %2, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xui16>, tensor<ui16>) -> tensor<2x4x6xui16>
    %4 = stablehlo.constant dense<0> : tensor<ui16>
    %5 = "stablehlo.select_and_scatter"(%3, %0#0, %4) ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %8 = stablehlo.compare  GE, %arg0, %arg1,  UNSIGNED : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
      stablehlo.return %8 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %8 = stablehlo.add %arg0, %arg1 : tensor<ui16>
      stablehlo.return %8 : tensor<ui16>
    }) {window_dimensions = dense<[1, 3, 1]> : tensor<3xi64>, window_strides = dense<[1, 2, 1]> : tensor<3xi64>} : (tensor<2x4x6xui16>, tensor<2x1x6xui16>, tensor<ui16>) -> tensor<2x4x6xui16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 4, 6]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x4x6xui16>) -> tensor<2x4x6xui16>
    %7 = stablehlo.custom_call @check.eq(%6, %1) : (tensor<2x4x6xui16>, tensor<2x4x6xui16>) -> tensor<i1>
    return %7 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x1x6xui16>, tensor<2x4x6xui16>) {
    %0 = stablehlo.constant dense<[[[2, 3, 1, 4, 3, 1]], [[3, 1, 3, 1, 0, 3]]]> : tensor<2x1x6xui16>
    %1 = stablehlo.constant dense<[[[0, 3, 2, 5, 3, 1], [4, 0, 1, 4, 3, 0], [3, 1, 0, 0, 3, 1], [4, 4, 1, 3, 4, 3]], [[1, 1, 1, 0, 0, 3], [3, 3, 1, 0, 3, 0], [3, 2, 3, 0, 3, 2], [3, 2, 0, 4, 0, 0]]]> : tensor<2x4x6xui16>
    return %0, %1 : tensor<2x1x6xui16>, tensor<2x4x6xui16>
  }
  func.func private @expected() -> tensor<2x4x6xui16> {
    %0 = stablehlo.constant dense<[[[0, 3, 1, 4, 3, 1], [2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 1, 0, 3], [3, 1, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0]]]> : tensor<2x4x6xui16>
    return %0 : tensor<2x4x6xui16>
  }
}

