// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x7xf16>
    %1 = call @expected() : () -> tensor<5x7xf16>
    %2 = stablehlo.reduce_precision %0, format = e8m23 : tensor<5x7xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<5x7xf16>, tensor<5x7xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x7xf16> {
    %0 = stablehlo.constant dense<[[-3.521480e+00, -5.710940e+00, -6.386710e+00, 9.835930e+00, -4.351560e+00, 1.839840e+00, 7.230460e+00], [8.867180e-01, -6.535150e+00, -2.812500e+00, 2.539060e+00, 1.145510e+00, -1.121090e+00, 7.800780e+00], [5.792960e+00, 2.587890e-01, -1.869140e+00, 5.468750e-01, -4.812010e-01, 2.277340e+00, -1.941410e+00], [1.438480e+00, 4.800420e-02, 1.478520e+00, -3.548830e+00, -2.447510e-02, 6.007810e+00, 1.301760e+00], [-3.261720e+00, -3.876950e+00, 6.010740e-01, -2.906250e+00, 3.744140e+00, -2.560550e+00, 1.351560e+00]]> : tensor<5x7xf16>
    return %0 : tensor<5x7xf16>
  }
  func.func private @expected() -> tensor<5x7xf16> {
    %0 = stablehlo.constant dense<[[-3.521480e+00, -5.710940e+00, -6.386710e+00, 9.835930e+00, -4.351560e+00, 1.839840e+00, 7.230460e+00], [8.867180e-01, -6.535150e+00, -2.812500e+00, 2.539060e+00, 1.145510e+00, -1.121090e+00, 7.800780e+00], [5.792960e+00, 2.587890e-01, -1.869140e+00, 5.468750e-01, -4.812010e-01, 2.277340e+00, -1.941410e+00], [1.438480e+00, 4.800420e-02, 1.478520e+00, -3.548830e+00, -2.447510e-02, 6.007810e+00, 1.301760e+00], [-3.261720e+00, -3.876950e+00, 6.010740e-01, -2.906250e+00, 3.744140e+00, -2.560550e+00, 1.351560e+00]]> : tensor<5x7xf16>
    return %0 : tensor<5x7xf16>
  }
}
