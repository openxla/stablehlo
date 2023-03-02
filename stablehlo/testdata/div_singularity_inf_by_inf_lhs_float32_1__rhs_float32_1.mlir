// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1xf32>, tensor<1xf32>)
    %1 = call @expected() : () -> tensor<1xf32>
    %2 = stablehlo.divide %0#0, %0#1 : tensor<1xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1xf32>, tensor<1xf32>) {
    %0 = stablehlo.constant dense<0x7F800000> : tensor<1xf32>
    %1 = stablehlo.constant dense<0x7F800000> : tensor<1xf32>
    return %0, %1 : tensor<1xf32>, tensor<1xf32>
  }
  func.func private @expected() -> tensor<1xf32> {
    %0 = stablehlo.constant dense<0xFFC00000> : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
