// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf16>
    %1 = call @expected() : () -> tensor<8x9xf16>
    %2 = call @cumprod(%0) : (tensor<8x9xf16>) -> tensor<8x9xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf16>, tensor<8x9xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf16> {
    %0 = stablehlo.constant dense<[[-1.232910e-01, -2.150390e+00, -5.454100e-01, 5.937500e-01, -3.130860e+00, 5.126950e-01, 2.966800e+00, -1.430660e+00, -3.815920e-01], [1.009770e+00, 4.550780e+00, -1.801760e+00, 1.297850e+00, -2.301030e-01, 3.654790e-01, 3.722660e+00, 2.083980e+00, -1.708980e+00], [1.523440e+00, -2.443360e+00, -3.068360e+00, 3.969730e-01, 4.300780e+00, -1.385740e+00, 1.069340e+00, -3.515630e+00, 2.802730e-01], [-5.105470e+00, -2.488280e+00, -3.029300e+00, -1.701170e+00, -1.479490e+00, 2.154300e+00, -3.927730e+00, -8.930660e-01, -1.958010e+00], [3.312990e-01, -1.903320e+00, -3.302730e+00, -1.305660e+00, 1.452150e+00, -1.098630e+00, -5.085940e+00, 2.246090e-01, 5.648800e-02], [1.705080e+00, 4.679690e+00, 1.846680e+00, 4.484380e+00, 4.372560e-01, -8.538820e-02, -3.156250e+00, -1.712890e+00, -1.436520e+00], [1.479490e+00, -2.593750e+00, -5.800780e+00, -1.708010e+00, -4.214840e+00, 1.795900e+00, 5.457030e+00, -2.236330e+00, 1.116210e+00], [-1.026370e+00, -7.226560e-01, 2.240230e+00, -2.085940e+00, -1.214840e+00, -4.085940e+00, 3.847660e+00, 5.122070e-01, -1.672850e+00]]> : tensor<8x9xf16>
    return %0 : tensor<8x9xf16>
  }
  func.func private @expected() -> tensor<8x9xf16> {
    %0 = stablehlo.constant dense<[[-1.232910e-01, -2.150390e+00, -5.454100e-01, 5.937500e-01, -3.130860e+00, 5.126950e-01, 2.966800e+00, -1.430660e+00, -3.815920e-01], [-1.245120e-01, -9.789060e+00, 9.829100e-01, 7.705080e-01, 7.202140e-01, 1.873780e-01, 1.104690e+01, -2.982420e+00, 6.523440e-01], [-1.896970e-01, 2.392190e+01, -3.015630e+00, 3.059080e-01, 3.097660e+00, -2.597660e-01, 1.181250e+01, 1.048440e+01, 1.828610e-01], [9.682610e-01, -5.950000e+01, 9.140620e+00, -5.205080e-01, -4.582030e+00, -5.590820e-01, -4.637500e+01, -9.367180e+00, -3.579100e-01], [3.208010e-01, 1.132500e+02, -3.018750e+01, 6.796880e-01, -6.652340e+00, 6.142580e-01, 2.358750e+02, -2.103520e+00, -2.021790e-02], [5.468750e-01, 5.300000e+02, -5.575000e+01, 3.046880e+00, -2.908200e+00, -5.245970e-02, -7.440000e+02, 3.603520e+00, 2.905270e-02], [8.090820e-01, -1.375000e+03, 3.235000e+02, -5.203130e+00, 1.225780e+01, -9.423820e-02, -4.060000e+03, -8.062500e+00, 3.244020e-02], [-8.305660e-01, 9.930000e+02, 7.245000e+02, 1.085940e+01, -1.489060e+01, 3.850100e-01, -1.563200e+04, -4.128910e+00, -5.426030e-02]]> : tensor<8x9xf16>
    return %0 : tensor<8x9xf16>
  }
  func.func private @cumprod(%arg0: tensor<8x9xf16>) -> tensor<8x9xf16> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<4x9xf16>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<4x9xf16>
    %2 = stablehlo.multiply %0, %1 : tensor<4x9xf16>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<2x9xf16>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<2x9xf16>
    %5 = stablehlo.multiply %3, %4 : tensor<2x9xf16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %8 = stablehlo.multiply %6, %7 : tensor<1x9xf16>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf16>) -> tensor<0x9xf16>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<0x9xf16>
    %11 = stablehlo.multiply %9, %10 : tensor<0x9xf16>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xf16>, tensor<0x9xf16>) -> tensor<1x9xf16>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf16>, tensor<f16>) -> tensor<2x9xf16>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf16>, tensor<f16>) -> tensor<2x9xf16>
    %18 = stablehlo.add %15, %17 : tensor<2x9xf16>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<1x9xf16>
    %21 = stablehlo.multiply %19, %20 : tensor<1x9xf16>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<1x9xf16>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<2x9xf16>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf16>, tensor<f16>) -> tensor<4x9xf16>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf16>, tensor<f16>) -> tensor<4x9xf16>
    %28 = stablehlo.add %25, %27 : tensor<4x9xf16>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<3x9xf16>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<3x9xf16>
    %31 = stablehlo.multiply %29, %30 : tensor<3x9xf16>
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
