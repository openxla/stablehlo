// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4xui8>
    %1 = call @expected() : () -> tensor<4xui8>
    %2 = stablehlo.popcnt %0 : tensor<4xui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4xui8>, tensor<4xui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4xui8> {
    %0 = stablehlo.constant dense<[255, 254, 0, 1]> : tensor<4xui8>
    return %0 : tensor<4xui8>
  }
  func.func private @expected() -> tensor<4xui8> {
    %0 = stablehlo.constant dense<[8, 7, 0, 1]> : tensor<4xui8>
    return %0 : tensor<4xui8>
  }
}
