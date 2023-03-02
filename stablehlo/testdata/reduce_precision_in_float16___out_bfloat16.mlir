// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<f16>
    %1 = call @expected() : () -> tensor<f16>
    %2 = stablehlo.reduce_precision %0, format = e8m7 : tensor<f16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<f16>, tensor<f16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<f16> {
    %0 = stablehlo.constant dense<-1.836910e+00> : tensor<f16>
    return %0 : tensor<f16>
  }
  func.func private @expected() -> tensor<f16> {
    %0 = stablehlo.constant dense<-1.835940e+00> : tensor<f16>
    return %0 : tensor<f16>
  }
}
