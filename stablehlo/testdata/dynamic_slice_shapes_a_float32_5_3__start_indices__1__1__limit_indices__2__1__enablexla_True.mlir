// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<5x3xf32>, tensor<2xi32>)
    %1 = call @expected() : () -> tensor<1x0xf32>
    %2 = "stablehlo.slice"(%0#1) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
    %4 = "stablehlo.slice"(%0#1) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %5 = stablehlo.reshape %4 : (tensor<1xi32>) -> tensor<i32>
    %6 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.compare  LT, %3, %6,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %8 = stablehlo.constant dense<5> : tensor<i32>
    %9 = stablehlo.add %3, %8 : tensor<i32>
    %10 = stablehlo.select %7, %9, %3 : tensor<i1>, tensor<i32>
    %11 = stablehlo.constant dense<0> : tensor<i32>
    %12 = stablehlo.compare  LT, %5, %11,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = stablehlo.constant dense<3> : tensor<i32>
    %14 = stablehlo.add %5, %13 : tensor<i32>
    %15 = stablehlo.select %12, %14, %5 : tensor<i1>, tensor<i32>
    %16 = stablehlo.dynamic_slice %0#0, %10, %15, sizes = [1, 0] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x0xf32>
    %17 = stablehlo.custom_call @check.eq(%16, %1) : (tensor<1x0xf32>, tensor<1x0xf32>) -> tensor<i1>
    return %17 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x3xf32>, tensor<2xi32>) {
    %0 = stablehlo.constant dense<[[5.77195787, -0.853386402, 0.736518919], [0.0973336771, -0.737931609, -2.76217175], [1.97758496, 0.0812616571, 1.65601397], [1.83973408, -7.90198993, -2.39591622], [0.935606241, 2.10476828, 0.343096346]]> : tensor<5x3xf32>
    %1 = stablehlo.constant dense<1> : tensor<2xi32>
    return %0, %1 : tensor<5x3xf32>, tensor<2xi32>
  }
  func.func private @expected() -> tensor<1x0xf32> {
    %0 = stablehlo.constant dense<> : tensor<1x0xf32>
    return %0 : tensor<1x0xf32>
  }
}
