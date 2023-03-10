// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x7xf16>
    %1 = call @expected() : () -> tensor<5x7xf16>
    %2 = stablehlo.reduce_precision %0, format = e5m10 : tensor<5x7xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<5x7xf16>, tensor<5x7xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x7xf16> {
    %0 = stablehlo.constant dense<[[1.535160e+00, -1.493160e+00, -3.308110e-01, 2.070310e+00, -2.009770e+00, 1.943970e-02, 2.194820e-01], [3.789060e+00, 1.041990e+00, -2.233890e-01, -1.038090e+00, -4.246090e+00, 3.634770e+00, -5.644530e+00], [-5.483400e-01, -1.176760e-01, -4.234380e+00, 2.792970e+00, 1.507570e-01, -6.289060e-01, -3.335940e+00], [-6.125000e+00, -7.085930e+00, 8.847650e-01, 1.594240e-01, -3.683590e+00, 2.433590e+00, -1.701170e+00], [-7.675780e+00, -1.316410e+00, -2.795410e-01, -3.810550e+00, 3.150390e+00, -1.585940e+00, 4.660160e+00]]> : tensor<5x7xf16>
    return %0 : tensor<5x7xf16>
  }
  func.func private @expected() -> tensor<5x7xf16> {
    %0 = stablehlo.constant dense<[[1.535160e+00, -1.493160e+00, -3.308110e-01, 2.070310e+00, -2.009770e+00, 1.943970e-02, 2.194820e-01], [3.789060e+00, 1.041990e+00, -2.233890e-01, -1.038090e+00, -4.246090e+00, 3.634770e+00, -5.644530e+00], [-5.483400e-01, -1.176760e-01, -4.234380e+00, 2.792970e+00, 1.507570e-01, -6.289060e-01, -3.335940e+00], [-6.125000e+00, -7.085930e+00, 8.847650e-01, 1.594240e-01, -3.683590e+00, 2.433590e+00, -1.701170e+00], [-7.675780e+00, -1.316410e+00, -2.795410e-01, -3.810550e+00, 3.150390e+00, -1.585940e+00, 4.660160e+00]]> : tensor<5x7xf16>
    return %0 : tensor<5x7xf16>
  }
}
