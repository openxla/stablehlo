// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<100x100xi1>
    %1 = call @expected() : () -> tensor<100x100xbf16>
    %2 = stablehlo.convert %0 : (tensor<100x100xi1>) -> tensor<100x100xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<100x100xbf16>, tensor<100x100xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<100x100xi1> {
    %0 = stablehlo.constant dense<true> : tensor<100x100xi1>
    return %0 : tensor<100x100xi1>
  }
  func.func private @expected() -> tensor<100x100xbf16> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<100x100xbf16>
    return %0 : tensor<100x100xbf16>
  }
}
