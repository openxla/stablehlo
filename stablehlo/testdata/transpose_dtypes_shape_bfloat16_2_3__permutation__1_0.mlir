// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xbf16>
    %1 = call @expected() : () -> tensor<3x2xbf16>
    %2 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xbf16>) -> tensor<3x2xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xbf16>, tensor<3x2xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[9.921870e-01, 8.828120e-01, -1.179690e+00], [1.726560e+00, -1.156250e+00, 7.656250e-01]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
  func.func private @expected() -> tensor<3x2xbf16> {
    %0 = stablehlo.constant dense<[[9.921870e-01, 1.726560e+00], [8.828120e-01, -1.156250e+00], [-1.179690e+00, 7.656250e-01]]> : tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}
