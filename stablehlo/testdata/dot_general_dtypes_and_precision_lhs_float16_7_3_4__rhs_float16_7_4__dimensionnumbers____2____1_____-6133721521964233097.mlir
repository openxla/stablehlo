// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xf16>, tensor<7x4xf16>)
    %1 = call @expected() : () -> tensor<7x3xf16>
    %2 = stablehlo.convert %0#0 : (tensor<7x3x4xf16>) -> tensor<7x3x4xf32>
    %3 = stablehlo.convert %0#1 : (tensor<7x4xf16>) -> tensor<7x4xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<7x3x4xf32>, tensor<7x4xf32>) -> tensor<7x3xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<7x3xf16>, tensor<7x3xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xf16>, tensor<7x4xf16>) {
    %0 = stablehlo.constant dense<[[[4.414060e-01, -6.449210e+00, -3.242190e+00, 4.195310e+00], [8.369140e-01, 1.888670e+00, 3.191410e+00, -2.817380e-01], [-7.202140e-01, 1.684570e+00, -2.498050e+00, -5.556640e-01]], [[-4.339840e+00, 1.483150e-01, 2.226560e+00, 3.251950e+00], [8.979490e-01, 1.440430e+00, 3.802730e+00, -3.605470e+00], [1.535160e+00, 4.294430e-01, -3.720700e+00, 4.523440e+00]], [[3.488280e+00, -6.613280e+00, 2.365230e+00, 3.451170e+00], [1.567380e+00, 5.039060e+00, -4.089840e+00, 6.597650e+00], [6.147460e-01, 3.427730e+00, -2.031250e+00, -9.697260e-01]], [[2.425780e+00, -5.673830e-01, 1.345700e+00, -5.961910e-01], [-3.656250e+00, 1.254880e+00, 1.595700e+00, -1.459960e+00], [-2.978520e+00, -1.188480e+00, -5.512700e-01, 5.590820e-01]], [[-1.822270e+00, -3.218750e+00, 3.523440e+00, 2.115230e+00], [9.257810e-01, -6.281250e+00, -3.562010e-01, -2.982420e+00], [3.197270e+00, -1.303710e+00, 7.031250e+00, -1.556400e-01]], [[2.082030e+00, 1.159180e+00, 1.652340e+00, -3.824220e+00], [-9.428710e-01, -5.851560e+00, 1.023440e+00, 2.691410e+00], [1.465820e+00, -6.699210e-01, -1.385740e+00, -8.549800e-01]], [[8.305660e-01, 5.906250e+00, 5.488280e+00, 2.822270e-01], [6.455080e-01, -1.617190e+00, 3.839840e+00, -4.605470e+00], [-2.666020e+00, 2.705080e-01, -4.757810e+00, -9.790030e-01]]]> : tensor<7x3x4xf16>
    %1 = stablehlo.constant dense<[[-6.109380e+00, -2.683590e+00, -2.800780e+00, -2.663570e-01], [-4.105470e+00, 9.672850e-01, -3.250000e+00, 3.267580e+00], [-4.575200e-01, 1.502930e+00, 2.816410e+00, -2.384770e+00], [3.093750e+00, 5.585940e-01, 3.654300e+00, -3.464840e+00], [1.245120e+00, -1.692380e+00, -1.093140e-01, -3.771970e-01], [-1.436520e+00, -2.472660e+00, -7.353520e-01, -5.531250e+00], [6.000000e+00, -1.603520e+00, -4.082030e-01, -3.247070e-01]]> : tensor<7x4xf16>
    return %0, %1 : tensor<7x3x4xf16>, tensor<7x4xf16>
  }
  func.func private @expected() -> tensor<7x3xf16> {
    %0 = stablehlo.constant dense<[[2.257810e+01, -1.904690e+01, 7.023430e+00], [2.134380e+01, -2.643750e+01, 2.098440e+01], [-1.310160e+01, -2.039060e+01, 1.461910e+00], [1.417190e+01, 2.790530e-01, -1.382810e+01], [1.995120e+00, 1.294530e+01, 5.476560e+00], [1.407810e+01, 1.839600e-01, 5.300780e+00], [-6.820310e+00, 6.394530e+00, -1.417190e+01]]> : tensor<7x3xf16>
    return %0 : tensor<7x3xf16>
  }
}
