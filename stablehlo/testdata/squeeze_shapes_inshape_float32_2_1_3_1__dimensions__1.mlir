// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x1x3x1xf32>
    %1 = call @expected() : () -> tensor<2x3x1xf32>
    %2 = stablehlo.reshape %0 : (tensor<2x1x3x1xf32>) -> tensor<2x3x1xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3x1xf32>, tensor<2x3x1xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x1x3x1xf32> {
    %0 = stablehlo.constant dense<[[[[1.5098201], [3.00185275], [-1.86794591]]], [[[2.27975774], [-0.793432116], [4.07953072]]]]> : tensor<2x1x3x1xf32>
    return %0 : tensor<2x1x3x1xf32>
  }
  func.func private @expected() -> tensor<2x3x1xf32> {
    %0 = stablehlo.constant dense<[[[1.5098201], [3.00185275], [-1.86794591]], [[2.27975774], [-0.793432116], [4.07953072]]]> : tensor<2x3x1xf32>
    return %0 : tensor<2x3x1xf32>
  }
}
