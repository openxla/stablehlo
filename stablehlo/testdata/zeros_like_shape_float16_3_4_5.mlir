// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4x5xf16>
    %1 = call @expected() : () -> tensor<3x4x5xf16>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f16>) -> tensor<3x4x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x4x5xf16>, tensor<3x4x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4x5xf16> {
    %0 = stablehlo.constant dense<[[[1.485350e+00, -1.055660e+00, -1.452150e+00, 3.751950e+00, 1.122070e+00], [-1.505860e+00, -5.172730e-02, 2.253420e-01, -7.354740e-02, -4.484380e+00], [6.577150e-01, -1.958980e+00, -1.197270e+00, -1.143550e+00, 3.691410e+00], [-1.774410e+00, -2.138670e+00, 3.415530e-01, 4.429690e+00, -4.570310e-01]], [[5.683590e+00, 1.069340e+00, -3.867190e-01, -2.308590e+00, 3.964840e+00], [-3.490230e+00, 3.398440e+00, -2.496090e+00, 2.464840e+00, 1.211910e+00], [5.816650e-02, -2.978520e+00, -2.785160e+00, 3.751950e+00, -1.932620e+00], [-3.888670e+00, 2.783200e-01, -1.246090e+00, -2.091800e+00, 2.130860e+00]], [[-9.536130e-01, -1.071290e+00, 3.480470e+00, 3.037110e+00, -2.224610e+00], [1.860350e+00, 2.669920e+00, 2.464840e+00, -1.510740e+00, -2.946780e-01], [-3.369140e+00, 2.074220e+00, 3.412110e+00, -3.982420e+00, 2.650390e+00], [-6.914060e-01, -2.857420e+00, -2.806640e+00, -4.214840e+00, -2.519530e+00]]]> : tensor<3x4x5xf16>
    return %0 : tensor<3x4x5xf16>
  }
  func.func private @expected() -> tensor<3x4x5xf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<3x4x5xf16>
    return %0 : tensor<3x4x5xf16>
  }
}
