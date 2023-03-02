// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x5xi16>
    %1 = call @expected() : () -> tensor<4x5xi16>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xi16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x5xi16>, tensor<4x5xi16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x5xi16> {
    %0 = stablehlo.constant dense<[[1, 0, 1, 3, 0], [0, 3, 0, 0, -2], [4, -3, 5, 0, 5], [4, -3, 6, -3, 0]]> : tensor<4x5xi16>
    return %0 : tensor<4x5xi16>
  }
  func.func private @expected() -> tensor<4x5xi16> {
    %0 = stablehlo.constant dense<[[4, -3, 6, -3, 0], [4, -3, 5, 0, 5], [0, 3, 0, 0, -2], [1, 0, 1, 3, 0]]> : tensor<4x5xi16>
    return %0 : tensor<4x5xi16>
  }
}
