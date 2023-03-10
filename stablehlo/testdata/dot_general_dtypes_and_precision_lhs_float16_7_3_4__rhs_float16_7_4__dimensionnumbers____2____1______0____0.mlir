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
    %0 = stablehlo.constant dense<[[[1.151370e+00, 6.472650e+00, 2.406250e+00, -2.878910e+00], [-3.070310e+00, 6.210940e-01, 4.421880e+00, -3.474610e+00], [-2.779300e+00, -4.167480e-01, 5.590820e-01, 5.765630e+00]], [[5.402340e+00, 2.087890e+00, -3.914060e+00, 5.992190e+00], [6.769530e+00, -1.975590e+00, -5.878900e-01, 6.376950e-01], [-3.945310e+00, -3.287110e+00, -1.043950e+00, 2.937500e+00]], [[5.639650e-01, 4.489750e-01, 3.264160e-01, 6.665040e-01], [3.681640e+00, 2.789060e+00, -2.187500e-01, 1.422850e+00], [1.595460e-01, -1.531250e+00, -2.693360e+00, -8.256830e-01]], [[9.799800e-01, 9.344480e-02, 5.765630e+00, -3.214840e+00], [1.758790e+00, -1.082030e+00, -6.609380e+00, 1.777340e+00], [8.940420e-01, -1.050780e+00, -2.380860e+00, 6.789060e+00]], [[3.791020e+00, -2.722660e+00, 3.294920e+00, -7.153320e-01], [8.208000e-01, -3.109380e+00, 6.023410e-03, -3.452150e-01], [-1.989750e-01, 5.796880e+00, -1.272460e+00, -6.069340e-01]], [[2.868650e-01, -6.235350e-01, -5.395510e-01, -2.730470e+00], [-5.855460e+00, 8.476560e-01, -3.437500e+00, -2.017580e+00], [-3.804690e+00, -1.536870e-01, 5.007810e+00, 2.142580e+00]], [[1.958980e+00, -1.810550e+00, 1.309570e+00, 4.242190e+00], [4.912110e-01, -1.571290e+00, 6.230460e+00, -2.402340e-01], [-5.359380e+00, -1.025000e+01, 9.111320e-01, -1.181640e+00]]]> : tensor<7x3x4xf16>
    %1 = stablehlo.constant dense<[[-2.326170e+00, -2.486330e+00, -2.347660e+00, -6.300780e+00], [-8.046880e+00, 8.876950e-01, -3.203130e+00, 1.947270e+00], [2.365230e+00, -6.699210e-01, -2.148440e+00, 2.921880e+00], [-3.050780e+00, 2.214840e+00, -4.144530e+00, -2.943360e+00], [-3.306640e+00, -1.476560e+00, 2.474610e+00, 1.367190e+00], [4.683590e+00, 3.763670e+00, 4.582030e+00, 1.289060e-01], [-1.448240e+00, -2.039060e+00, 6.781250e+00, -3.632810e+00]]> : tensor<7x4xf16>
    return %0, %1 : tensor<7x3x4xf16>, tensor<7x4xf16>
  }
  func.func private @expected() -> tensor<7x3xf16> {
    %0 = stablehlo.constant dense<[[-6.281250e+00, 1.710940e+01, -3.014060e+01], [-1.740630e+01, -5.309380e+01, 3.790630e+01], [2.279300e+00, 1.146880e+01, 4.777340e+00], [-1.721880e+01, 1.439840e+01, -1.517190e+01], [-1.339840e+00, 1.419920e+00, -1.188280e+01], [-3.828130e+00, -4.025000e+01, 4.824220e+00], [-5.675780e+00, 4.562500e+01, 3.912500e+01]]> : tensor<7x3xf16>
    return %0 : tensor<7x3xf16>
  }
}

