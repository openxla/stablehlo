// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.clamp %0#0, %0#1, %0#2 : tensor<2x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = stablehlo.constant dense<[[0.0898445472, -5.46463346, -0.46935451], [-0.777897596, 3.82473087, 1.76897967]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<[[-2.05830979, 4.15932322, -2.54856324], [2.31949759, 1.92253649, -2.71293783]]> : tensor<2x3xf32>
    %2 = stablehlo.constant dense<[[-3.24641538, -5.525650e+00, -1.07148051], [-0.465055346, 4.45410252, 0.272830278]]> : tensor<2x3xf32>
    return %0, %1, %2 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[-3.24641538, -5.525650e+00, -1.07148051], [-0.465055346, 3.82473087, 0.272830278]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
