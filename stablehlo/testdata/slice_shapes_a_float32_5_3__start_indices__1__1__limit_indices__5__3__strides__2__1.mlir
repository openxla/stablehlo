// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x3xf32>
    %1 = call @expected() : () -> tensor<2x2xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<[5, 3]> : tensor<2xi64>, start_indices = dense<1> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<5x3xf32>) -> tensor<2x2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x3xf32> {
    %0 = stablehlo.constant dense<[[-4.05739403, 1.44734287, 1.84830022], [-0.606142938, 1.1464169, -0.033899311], [-3.79691529, 3.13463449, 1.86898649], [1.96301615, 3.3724997, -0.299463451], [-3.28946066, -1.63501847, -1.46235979]]> : tensor<5x3xf32>
    return %0 : tensor<5x3xf32>
  }
  func.func private @expected() -> tensor<2x2xf32> {
    %0 = stablehlo.constant dense<[[1.1464169, -0.033899311], [3.3724997, -0.299463451]]> : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
