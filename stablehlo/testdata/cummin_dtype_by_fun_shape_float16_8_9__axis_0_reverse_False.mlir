// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf16>
    %1 = call @expected() : () -> tensor<8x9xf16>
    %2 = call @cummin(%0) : (tensor<8x9xf16>) -> tensor<8x9xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf16>, tensor<8x9xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf16> {
    %0 = stablehlo.constant dense<[[-4.855470e+00, -3.644530e+00, 4.957030e+00, 2.107420e+00, -3.511720e+00, 6.683590e+00, -2.474610e+00, 2.578130e+00, 4.860840e-01], [-1.002930e+00, -1.990230e+00, -4.011720e+00, -2.917970e+00, 2.095700e+00, 1.721680e+00, 4.402340e+00, 1.124020e+00, -1.566410e+00], [6.054690e-01, 3.435060e-01, 2.853520e+00, 5.439450e-01, 1.297850e+00, 2.675780e+00, 1.101560e+00, 2.390630e+00, 3.167970e+00], [3.105470e+00, 1.671880e+00, 5.308590e+00, -1.698240e+00, -5.498050e-01, -4.428710e-01, 2.587890e+00, -3.544920e+00, -2.720700e+00], [-4.406250e+00, 1.946290e+00, -2.623050e+00, -1.006840e+00, 1.855470e+00, 9.399410e-01, -2.173830e+00, -2.599610e+00, -2.328130e+00], [-2.005860e+00, -8.144530e-01, -3.205570e-01, 8.203130e-01, -5.341800e-01, -8.281250e-01, -1.211910e+00, 1.993160e+00, -2.087890e+00], [3.765630e+00, -1.872070e+00, -6.484380e-01, 1.858400e+00, 1.001950e+00, -4.853520e-01, -5.320310e+00, 1.103520e+00, 1.082030e+00], [3.134770e+00, 2.654300e+00, -4.671880e+00, 8.168940e-01, 3.849610e+00, -1.726070e-01, -3.959960e-01, -3.234380e+00, -1.875980e+00]]> : tensor<8x9xf16>
    return %0 : tensor<8x9xf16>
  }
  func.func private @expected() -> tensor<8x9xf16> {
    %0 = stablehlo.constant dense<[[-4.855470e+00, -3.644530e+00, 4.957030e+00, 2.107420e+00, -3.511720e+00, 6.683590e+00, -2.474610e+00, 2.578130e+00, 4.860840e-01], [-4.855470e+00, -3.644530e+00, -4.011720e+00, -2.917970e+00, -3.511720e+00, 1.721680e+00, -2.474610e+00, 1.124020e+00, -1.566410e+00], [-4.855470e+00, -3.644530e+00, -4.011720e+00, -2.917970e+00, -3.511720e+00, 1.721680e+00, -2.474610e+00, 1.124020e+00, -1.566410e+00], [-4.855470e+00, -3.644530e+00, -4.011720e+00, -2.917970e+00, -3.511720e+00, -4.428710e-01, -2.474610e+00, -3.544920e+00, -2.720700e+00], [-4.855470e+00, -3.644530e+00, -4.011720e+00, -2.917970e+00, -3.511720e+00, -4.428710e-01, -2.474610e+00, -3.544920e+00, -2.720700e+00], [-4.855470e+00, -3.644530e+00, -4.011720e+00, -2.917970e+00, -3.511720e+00, -8.281250e-01, -2.474610e+00, -3.544920e+00, -2.720700e+00], [-4.855470e+00, -3.644530e+00, -4.011720e+00, -2.917970e+00, -3.511720e+00, -8.281250e-01, -5.320310e+00, -3.544920e+00, -2.720700e+00], [-4.855470e+00, -3.644530e+00, -4.671880e+00, -2.917970e+00, -3.511720e+00, -8.281250e-01, -5.320310e+00, -3.544920e+00, -2.720700e+00]]> : tensor<8x9xf16>
    return %0 : tensor<8x9xf16>
  }
  func.func private @cummin(%arg0: tensor<8x9xf16>) -> tensor<8x9xf16> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<4x9xf16>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<4x9xf16>
    %2 = stablehlo.minimum %0, %1 : tensor<4x9xf16>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<2x9xf16>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<2x9xf16>
    %5 = stablehlo.minimum %3, %4 : tensor<2x9xf16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %8 = stablehlo.minimum %6, %7 : tensor<1x9xf16>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf16>) -> tensor<0x9xf16>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<0x9xf16>
    %11 = stablehlo.minimum %9, %10 : tensor<0x9xf16>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xf16>, tensor<0x9xf16>) -> tensor<1x9xf16>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf16>, tensor<f16>) -> tensor<2x9xf16>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf16>, tensor<f16>) -> tensor<2x9xf16>
    %18 = stablehlo.add %15, %17 : tensor<2x9xf16>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<1x9xf16>
    %21 = stablehlo.minimum %19, %20 : tensor<1x9xf16>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<1x9xf16>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<2x9xf16>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf16>, tensor<f16>) -> tensor<4x9xf16>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf16>, tensor<f16>) -> tensor<4x9xf16>
    %28 = stablehlo.add %25, %27 : tensor<4x9xf16>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<3x9xf16>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<3x9xf16>
    %31 = stablehlo.minimum %29, %30 : tensor<3x9xf16>
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
