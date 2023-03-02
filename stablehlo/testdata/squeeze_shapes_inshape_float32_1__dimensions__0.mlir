// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<1xf32>
    %1 = call @expected() : () -> tensor<f32>
    %2 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<1xf32> {
    %0 = stablehlo.constant dense<-5.2086997> : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  func.func private @expected() -> tensor<f32> {
    %0 = stablehlo.constant dense<-5.2086997> : tensor<f32>
    return %0 : tensor<f32>
  }
}
