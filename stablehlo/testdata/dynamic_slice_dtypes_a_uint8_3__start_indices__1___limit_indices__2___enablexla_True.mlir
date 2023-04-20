// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xui8>, tensor<1xi32>)
    %1 = call @expected() : () -> tensor<1xui8>
    %2 = "stablehlo.slice"(%0#1) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
    %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.compare  LT, %3, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = stablehlo.constant dense<3> : tensor<i32>
    %7 = stablehlo.add %3, %6 : tensor<i32>
    %8 = stablehlo.select %5, %7, %3 : tensor<i1>, tensor<i32>
    %9 = stablehlo.dynamic_slice %0#0, %8, sizes = [1] : (tensor<3xui8>, tensor<i32>) -> tensor<1xui8>
    %10 = stablehlo.custom_call @check.eq(%9, %1) : (tensor<1xui8>, tensor<1xui8>) -> tensor<i1>
    return %10 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xui8>, tensor<1xi32>) {
    %0 = stablehlo.constant dense<[4, 5, 0]> : tensor<3xui8>
    %1 = stablehlo.constant dense<1> : tensor<1xi32>
    return %0, %1 : tensor<3xui8>, tensor<1xi32>
  }
  func.func private @expected() -> tensor<1xui8> {
    %0 = stablehlo.constant dense<5> : tensor<1xui8>
    return %0 : tensor<1xui8>
  }
}
