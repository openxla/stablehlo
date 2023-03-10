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
    %0 = stablehlo.constant dense<[[1.744140e+00, -2.251950e+00, 2.662110e+00, -7.543940e-01, 2.162110e+00, -3.266600e-01, 3.361330e+00, 8.078130e+00, -7.402340e-01], [2.605470e+00, -3.662110e+00, -3.921880e+00, -5.288700e-02, 2.988280e+00, -4.816890e-01, 2.046880e+00, 5.578130e+00, 5.717770e-01], [7.433590e+00, -1.799320e-01, 1.553710e+00, -2.214840e+00, 1.310550e+00, -3.437500e+00, 2.785160e+00, -9.819330e-01, -5.187500e+00], [-5.558590e+00, 1.310550e+00, 2.601560e+00, -3.791500e-01, 2.931640e+00, -3.136720e+00, 3.277340e+00, 1.157230e+00, -2.898440e+00], [-2.863280e+00, -6.816400e-01, 3.662110e+00, 1.871090e+00, -2.919920e+00, -5.503900e+00, 2.946470e-02, 1.721680e+00, -8.461910e-01], [9.360350e-01, 4.001460e-01, 6.181640e-01, 2.335940e+00, 3.265630e+00, 4.755860e-01, 1.258790e+00, 2.167970e+00, 1.049800e+00], [5.085940e+00, 6.148440e+00, 3.080080e+00, 1.196290e+00, -6.109380e+00, -7.234380e+00, 2.869140e+00, 5.258790e-01, 2.580080e+00], [-2.166020e+00, 1.349610e+00, -5.171880e+00, -1.611330e+00, -4.257810e+00, 1.674800e+00, -1.120120e+00, -2.388670e+00, -1.922850e+00]]> : tensor<8x9xf16>
    return %0 : tensor<8x9xf16>
  }
  func.func private @expected() -> tensor<8x9xf16> {
    %0 = stablehlo.constant dense<[[1.744140e+00, -2.251950e+00, 2.662110e+00, -7.543940e-01, 2.162110e+00, -3.266600e-01, 3.361330e+00, 8.078130e+00, -7.402340e-01], [4.542970e+00, 8.250000e+00, -1.043750e+01, 3.988650e-02, 6.460940e+00, 1.573490e-01, 6.878900e+00, 4.506250e+01, -4.233400e-01], [3.378130e+01, -1.484380e+00, -1.621880e+01, -8.831780e-02, 8.468750e+00, -5.410150e-01, 1.915630e+01, -4.425000e+01, 2.195310e+00], [-1.876250e+02, -1.945310e+00, -4.218750e+01, 3.350830e-02, 2.482810e+01, 1.696290e+00, 6.278130e+01, -5.121880e+01, -6.367190e+00], [5.370000e+02, 1.326170e+00, -1.545000e+02, 6.268310e-02, -7.250000e+01, -9.335930e+00, 1.849610e+00, -8.818750e+01, 5.386710e+00], [5.027500e+02, 5.302730e-01, -9.550000e+01, 1.464840e-01, -2.368750e+02, -4.441410e+00, 2.328130e+00, -1.911250e+02, 5.656250e+00], [2.556000e+03, 3.259770e+00, -2.942500e+02, 1.752930e-01, 1.447000e+03, 3.212500e+01, 6.679690e+00, -1.005000e+02, 1.459380e+01], [-5.536000e+03, 4.398440e+00, 1.521000e+03, -2.824710e-01, -6.160000e+03, 5.381250e+01, -7.480460e+00, 2.401250e+02, -2.806250e+01]]> : tensor<8x9xf16>
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
