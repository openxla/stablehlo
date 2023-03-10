// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x5xf32>
    %1 = call @expected() : () -> tensor<4x5xf32>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x5xf32> {
    %0 = stablehlo.constant dense<[[-6.02407265, 3.11093283, -2.532550e+00, -4.78078699, 2.53503942], [1.270130e+00, -1.59578395, 3.28086519, -1.46065211, -4.57731438], [-0.32994166, 3.34717584, 2.85873795, -3.4305625, 1.44537222], [-0.922935426, -4.55431795, 6.62983989, 3.72943854, 5.40398884]]> : tensor<4x5xf32>
    return %0 : tensor<4x5xf32>
  }
  func.func private @expected() -> tensor<4x5xf32> {
    %0 = stablehlo.constant dense<[[-0.922935426, -4.55431795, 6.62983989, 3.72943854, 5.40398884], [-0.32994166, 3.34717584, 2.85873795, -3.4305625, 1.44537222], [1.270130e+00, -1.59578395, 3.28086519, -1.46065211, -4.57731438], [-6.02407265, 3.11093283, -2.532550e+00, -4.78078699, 2.53503942]]> : tensor<4x5xf32>
    return %0 : tensor<4x5xf32>
  }
}
