// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf16>
    %1 = call @expected() : () -> tensor<8x9xf16>
    %2 = call @cumsum(%0) : (tensor<8x9xf16>) -> tensor<8x9xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf16>, tensor<8x9xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf16> {
    %0 = stablehlo.constant dense<[[4.191890e-01, 3.533200e+00, -4.007810e+00, -1.465820e+00, -7.841800e-01, 5.441400e+00, 2.550780e+00, 1.158200e+00, -9.384760e-01], [-1.325200e+00, -1.642580e+00, 5.039060e+00, -4.453130e+00, 1.210940e+00, -2.277830e-01, -1.099610e+00, -4.097660e+00, -5.888670e-01], [-3.912110e+00, 7.011710e-01, 4.640630e+00, -1.728520e+00, 7.963860e-01, -2.046880e+00, 2.507810e+00, 9.820550e-02, 3.291020e+00], [3.326170e+00, 3.769530e+00, -4.207030e+00, -1.450200e+00, -3.388670e+00, -5.964840e+00, -5.382810e+00, 1.232420e+00, -1.415040e+00], [-1.574220e+00, -5.964840e+00, -6.323240e-02, -6.582030e-01, 3.570310e+00, -1.573240e+00, -8.666990e-01, -8.471670e-01, -4.585940e+00], [-3.171880e+00, -3.122560e-01, -3.966800e+00, 3.503910e+00, -1.938480e+00, -6.772460e-01, -2.740230e+00, 6.566400e+00, -2.527340e+00], [-3.443360e+00, -3.436280e-02, -1.147460e+00, 4.714840e+00, 2.281250e+00, -1.322270e+00, -1.494140e+00, -1.947270e+00, 3.128910e+00], [-2.834470e-01, 1.401370e+00, -7.207030e+00, -3.570310e+00, -1.961910e+00, -2.833980e+00, -2.033200e+00, 7.153320e-01, -7.320310e+00]]> : tensor<8x9xf16>
    return %0 : tensor<8x9xf16>
  }
  func.func private @expected() -> tensor<8x9xf16> {
    %0 = stablehlo.constant dense<[[4.191890e-01, 3.533200e+00, -4.007810e+00, -1.465820e+00, -7.841800e-01, 5.441400e+00, 2.550780e+00, 1.158200e+00, -9.384760e-01], [-9.062500e-01, 1.890630e+00, 1.031250e+00, -5.917960e+00, 4.267580e-01, 5.214840e+00, 1.451170e+00, -2.939450e+00, -1.527340e+00], [-4.820310e+00, 2.591800e+00, 5.671880e+00, -7.648430e+00, 1.222660e+00, 3.167970e+00, 3.958980e+00, -2.841800e+00, 1.763670e+00], [-1.492190e+00, 6.359380e+00, 1.464840e+00, -9.093750e+00, -2.164060e+00, -2.800780e+00, -1.423830e+00, -1.608400e+00, 3.486330e-01], [-3.066410e+00, 3.945310e-01, 1.401370e+00, -9.750000e+00, 1.406250e+00, -4.375000e+00, -2.291020e+00, -2.455080e+00, -4.238280e+00], [-6.238280e+00, 8.203130e-02, -2.566410e+00, -6.250000e+00, -5.322270e-01, -5.050780e+00, -5.031250e+00, 4.109380e+00, -6.765630e+00], [-9.679680e+00, 4.766850e-02, -3.714840e+00, -1.535160e+00, 1.749020e+00, -6.375000e+00, -6.523440e+00, 2.162110e+00, -3.636720e+00], [-9.960930e+00, 1.449220e+00, -1.092190e+01, -5.101560e+00, -2.128910e-01, -9.203120e+00, -8.554680e+00, 2.875000e+00, -1.095310e+01]]> : tensor<8x9xf16>
    return %0 : tensor<8x9xf16>
  }
  func.func private @cumsum(%arg0: tensor<8x9xf16>) -> tensor<8x9xf16> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<4x9xf16>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<4x9xf16>
    %2 = stablehlo.add %0, %1 : tensor<4x9xf16>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<2x9xf16>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<2x9xf16>
    %5 = stablehlo.add %3, %4 : tensor<2x9xf16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %8 = stablehlo.add %6, %7 : tensor<1x9xf16>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf16>) -> tensor<0x9xf16>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<0x9xf16>
    %11 = stablehlo.add %9, %10 : tensor<0x9xf16>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xf16>, tensor<0x9xf16>) -> tensor<1x9xf16>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf16>, tensor<f16>) -> tensor<2x9xf16>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf16>, tensor<f16>) -> tensor<2x9xf16>
    %18 = stablehlo.add %15, %17 : tensor<2x9xf16>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<1x9xf16>
    %21 = stablehlo.add %19, %20 : tensor<1x9xf16>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<1x9xf16>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<2x9xf16>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf16>, tensor<f16>) -> tensor<4x9xf16>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf16>, tensor<f16>) -> tensor<4x9xf16>
    %28 = stablehlo.add %25, %27 : tensor<4x9xf16>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<3x9xf16>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<3x9xf16>
    %31 = stablehlo.add %29, %30 : tensor<3x9xf16>
    %32 = "stablehlo.slice"(%arg0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<1x9xf16>
    %33 = stablehlo.concatenate %32, %31, dim = 0 : (tensor<1x9xf16>, tensor<3x9xf16>) -> tensor<4x9xf16>
    %34 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %35 = stablehlo.pad %33, %34, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xf16>, tensor<f16>) -> tensor<8x9xf16>
    %36 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %37 = stablehlo.pad %28, %36, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xf16>, tensor<f16>) -> tensor<8x9xf16>
    %38 = stablehlo.add %35, %37 : tensor<8x9xf16>
    return %38 : tensor<8x9xf16>
  }
}
