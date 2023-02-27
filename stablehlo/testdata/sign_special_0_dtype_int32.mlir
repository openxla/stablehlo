module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<2x2xi32>
    %1 = call @expected() : () -> tensor<2x2xi32>
    %2 = stablehlo.sign %0 : tensor<2x2xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @expected() -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<0> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
