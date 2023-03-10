// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xf16>, tensor<7x4xf16>)
    %1 = call @expected() : () -> tensor<7x3xf16>
    %2 = stablehlo.convert %0#0 : (tensor<7x3x4xf16>) -> tensor<7x3x4xf32>
    %3 = stablehlo.convert %0#1 : (tensor<7x4xf16>) -> tensor<7x4xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<7x3x4xf32>, tensor<7x4xf32>) -> tensor<7x3xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<7x3xf16>, tensor<7x3xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xf16>, tensor<7x4xf16>) {
    %0 = stablehlo.constant dense<[[[-7.753900e-01, -7.082030e+00, -1.917720e-01, 8.593750e-01], [6.240230e-01, -1.680660e+00, 7.914060e+00, 6.161500e-02], [3.078130e+00, 8.378900e-01, 2.531250e+00, -1.936520e+00]], [[-6.011710e+00, 5.683590e+00, -1.151370e+00, -3.593750e+00], [-3.117190e+00, 7.167960e-01, -4.226560e+00, 3.951170e+00], [3.429690e+00, -6.242190e+00, 2.035160e+00, 5.679690e+00]], [[4.675780e+00, -3.169920e+00, -7.753900e-01, -1.285160e+00], [4.074220e+00, -3.740230e-01, -2.408200e+00, 1.602540e+00], [-2.046880e+00, 3.101560e+00, 7.944330e-01, 1.098630e-01]], [[-2.771480e+00, -1.933590e-01, -3.251950e-01, -9.580070e-01], [6.313480e-01, 2.663570e-01, 3.291020e+00, 2.554690e+00], [-1.010740e+00, 1.131840e+00, 4.599610e-01, 9.421870e+00]], [[1.850590e+00, -4.074220e+00, -3.972660e+00, 4.160160e-01], [1.457520e-01, 2.904300e+00, -2.917970e+00, -2.080080e+00], [-5.054690e+00, 1.741210e+00, 4.457030e+00, 2.980470e+00]], [[1.628910e+00, -5.421880e+00, 2.296880e+00, -5.027340e+00], [5.515630e+00, 5.864260e-01, 4.218750e+00, 2.777340e+00], [-1.109380e+00, -3.537110e+00, 1.297850e+00, 4.406250e+00]], [[4.765630e+00, 5.542960e+00, -2.640630e+00, 1.156250e+00], [-2.193360e+00, -2.726560e+00, 1.460940e+00, 5.458980e-01], [-2.566410e+00, -1.573240e+00, 7.592770e-01, -1.068360e+00]]]> : tensor<7x3x4xf16>
    %1 = stablehlo.constant dense<[[3.310550e+00, 1.883790e+00, 1.255860e+00, 3.322750e-01], [9.892570e-01, -5.714840e+00, -1.817380e+00, 7.128900e-01], [-2.273440e+00, 6.023440e+00, 8.398430e-02, 3.390630e+00], [5.737300e-01, 1.044920e+00, 3.024900e-01, 7.792960e-01], [-2.703130e+00, 4.199220e+00, 2.287110e+00, -5.453130e+00], [-6.503900e+00, 1.004880e+00, 1.121090e+00, -2.539060e+00], [-1.180660e+00, -5.511710e+00, -4.046880e+00, 6.060790e-02]]> : tensor<7x4xf16>
    return %0, %1 : tensor<7x3x4xf16>, tensor<7x4xf16>
  }
  func.func private @expected() -> tensor<7x3xf16> {
    %0 = stablehlo.constant dense<[[-1.586720e+01, 8.859370e+00, 1.430470e+01], [-3.890630e+01, 3.318360e+00, 3.940630e+01], [-3.415630e+01, -6.285150e+00, 2.378130e+01], [-2.636720e+00, 3.626950e+00, 8.085930e+00], [-3.346880e+01, 1.646880e+01, 1.491410e+01], [-7.026360e-01, -3.759380e+01, -6.070310e+00], [-2.542190e+01, 1.174220e+01, 8.562500e+00]]> : tensor<7x3xf16>
    return %0 : tensor<7x3xf16>
  }
}

