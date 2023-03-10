// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4xf32>
    %1 = call @expected() : () -> tensor<3x4xf32>
    %2 = stablehlo.custom_call @check.eq(%0, %1) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<i1>
    return %2 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4xf32> {
    %0 = stablehlo.constant dense<[[2.97486401, 2.90809608, -0.295335233, -1.82598794], [-1.49391913, -3.06779861, -3.26420927, 0.995229721], [7.44611263, -3.13007045, -1.058210e+00, 1.50531471]]> : tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
  func.func private @expected() -> tensor<3x4xf32> {
    %0 = stablehlo.constant dense<[[2.97486401, 2.90809608, -0.295335233, -1.82598794], [-1.49391913, -3.06779861, -3.26420927, 0.995229721], [7.44611263, -3.13007045, -1.058210e+00, 1.50531471]]> : tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}
