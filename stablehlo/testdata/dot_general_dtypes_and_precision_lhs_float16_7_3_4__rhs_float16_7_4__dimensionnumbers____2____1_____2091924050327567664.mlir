// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xf16>, tensor<7x4xf16>)
    %1 = call @expected() : () -> tensor<7x3xf16>
    %2 = stablehlo.convert %0#0 : (tensor<7x3x4xf16>) -> tensor<7x3x4xf32>
    %3 = stablehlo.convert %0#1 : (tensor<7x4xf16>) -> tensor<7x4xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<7x3x4xf32>, tensor<7x4xf32>) -> tensor<7x3xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<7x3xf16>, tensor<7x3xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xf16>, tensor<7x4xf16>) {
    %0 = stablehlo.constant dense<[[[-3.675780e+00, -3.396480e+00, -3.349610e+00, -4.671880e+00], [8.862300e-01, 1.835940e+00, 1.183470e-01, -2.417970e+00], [-6.420900e-01, 1.375980e+00, -3.617190e+00, -4.250000e+00]], [[1.208010e+00, -1.507810e+00, 1.760740e+00, 1.792970e+00], [-3.537600e-01, 2.478520e+00, -3.326170e+00, 2.470700e-01], [1.353520e+00, 2.542970e+00, -1.622310e-01, -3.615230e+00]], [[2.250000e+00, 3.869140e+00, -1.359380e+00, 3.240230e+00], [1.142580e+00, -1.370120e+00, 4.699710e-01, -2.560550e+00], [-6.884770e-01, -7.036130e-01, -2.246090e+00, -4.804690e+00]], [[-3.503910e+00, -2.197270e+00, -1.815430e+00, 2.986330e+00], [-3.314210e-02, -5.034180e-01, 1.831050e+00, 2.222660e+00], [-5.226560e+00, -4.785160e-02, 1.060550e+00, 4.476560e+00]], [[5.527340e+00, 3.666020e+00, 5.566400e+00, -6.136710e+00], [1.406250e+00, -1.054690e+00, -1.380860e+00, 5.981450e-01], [-3.250000e+00, 8.023430e+00, -2.740230e+00, 2.781250e+00]], [[2.183590e+00, 9.788510e-03, 1.864010e-01, 9.218750e-01], [-1.898440e+00, 6.003900e+00, 5.693360e-01, -1.749020e+00], [3.429690e+00, 8.012690e-01, -2.187500e+00, -2.261720e+00]], [[3.003910e+00, -1.096680e+00, 3.708500e-01, -3.312500e+00], [-3.072270e+00, 7.128900e+00, -2.978520e+00, 7.778320e-01], [8.520500e-01, 4.140630e+00, -1.214840e+00, -7.458490e-02]]]> : tensor<7x3x4xf16>
    %1 = stablehlo.constant dense<[[4.629520e-02, -9.937500e+00, 3.802730e+00, 5.371090e-02], [3.787110e+00, -6.752930e-01, -8.171880e+00, 3.474120e-01], [8.129880e-01, -9.858390e-01, 1.416990e+00, 5.625000e+00], [-1.194340e+00, 5.597650e+00, -8.139640e-01, 3.546880e+00], [-1.904300e+00, -3.087890e+00, -3.052730e+00, 1.177730e+00], [1.861330e+00, 7.238280e+00, 3.048830e+00, -2.700200e-01], [3.195310e+00, -1.072270e+00, 2.728520e+00, 5.707030e+00]]> : tensor<7x4xf16>
    return %0, %1 : tensor<7x3x4xf16>, tensor<7x4xf16>
  }
  func.func private @expected() -> tensor<7x3xf16> {
    %0 = stablehlo.constant dense<[[2.059380e+01, -1.789060e+01, -2.768750e+01], [-8.171880e+00, 2.425000e+01, 3.478520e+00], [1.431250e+01, -1.146090e+01, -3.007810e+01], [3.955080e+00, 3.615230e+00, 2.098440e+01], [-4.606250e+01, 5.500000e+00, -6.945310e+00], [4.453130e+00, 4.212500e+01, 6.125000e+00], [-7.117180e+00, -2.115630e+01, -5.457030e+00]]> : tensor<7x3xf16>
    return %0 : tensor<7x3xf16>
  }
}
