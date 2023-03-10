// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xbf16>
    %1 = call @expected() : () -> tensor<8x9xbf16>
    %2 = call @cumprod(%0) : (tensor<8x9xbf16>) -> tensor<8x9xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xbf16>, tensor<8x9xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xbf16> {
    %0 = stablehlo.constant dense<[[2.546880e+00, -5.937500e-01, 2.421880e-01, 7.125000e+00, 3.046880e+00, 5.562500e+00, 9.101560e-01, 4.550780e-01, -1.429690e+00], [-5.187500e+00, 7.625000e+00, -2.843750e+00, -1.265630e+00, -1.748050e-01, -4.296880e-01, -1.734380e+00, -6.835930e-01, -2.078130e+00], [2.421880e+00, 3.343750e+00, 3.609380e+00, 3.759770e-02, 3.906250e+00, 2.703130e+00, 1.484380e+00, 4.218750e+00, 2.484380e+00], [3.295900e-02, -2.750000e+00, 1.929690e+00, -1.609380e+00, 4.125000e+00, -7.275390e-02, 3.140630e+00, -2.640630e+00, -2.359380e+00], [-1.828130e+00, 1.710940e+00, -7.031250e+00, -2.765630e+00, -1.531250e+00, 3.187500e+00, 7.578130e-01, 3.250000e+00, 6.000000e+00], [1.171880e+00, -2.468750e+00, 2.468750e+00, 3.687500e+00, 4.433590e-01, 2.000000e+00, -5.585940e-01, 4.718750e+00, 3.734380e+00], [-3.734380e+00, -1.671880e+00, -4.843750e+00, 1.218750e+00, 3.710940e-02, -2.828130e+00, 1.453130e+00, 2.437500e+00, 7.460930e-01], [-3.890630e+00, 4.375000e+00, -2.640630e+00, -2.875000e+00, 4.875000e+00, -1.367190e+00, -6.289060e-01, -2.859380e+00, -3.984380e+00]]> : tensor<8x9xbf16>
    return %0 : tensor<8x9xbf16>
  }
  func.func private @expected() -> tensor<8x9xbf16> {
    %0 = stablehlo.constant dense<[[2.546880e+00, -5.937500e-01, 2.421880e-01, 7.125000e+00, 3.046880e+00, 5.562500e+00, 9.101560e-01, 4.550780e-01, -1.429690e+00], [-1.318750e+01, -4.531250e+00, -6.875000e-01, -9.000000e+00, -5.312500e-01, -2.390630e+00, -1.578130e+00, -3.105470e-01, 2.968750e+00], [-3.200000e+01, -1.512500e+01, -2.484380e+00, -3.378910e-01, -2.078130e+00, -6.468750e+00, -2.343750e+00, -1.312500e+00, 7.375000e+00], [-1.046880e+00, 4.175000e+01, -4.781250e+00, 5.468750e-01, -8.562500e+00, 4.687500e-01, -7.343750e+00, 3.453130e+00, -1.750000e+01], [1.914060e+00, 7.150000e+01, 3.350000e+01, -1.515630e+00, 1.312500e+01, 1.492190e+00, -5.562500e+00, 1.125000e+01, -1.050000e+02], [2.234380e+00, -1.760000e+02, 8.300000e+01, -5.562500e+00, 5.812500e+00, 2.984380e+00, 3.109380e+00, 5.300000e+01, -3.920000e+02], [-8.375000e+00, 2.940000e+02, -4.020000e+02, -6.781250e+00, 2.158200e-01, -8.437500e+00, 4.531250e+00, 1.290000e+02, -2.920000e+02], [3.250000e+01, 1.288000e+03, 1.064000e+03, 1.950000e+01, 1.046880e+00, 1.156250e+01, -2.843750e+00, -3.680000e+02, 1.160000e+03]]> : tensor<8x9xbf16>
    return %0 : tensor<8x9xbf16>
  }
  func.func private @cumprod(%arg0: tensor<8x9xbf16>) -> tensor<8x9xbf16> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xbf16>) -> tensor<4x9xbf16>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xbf16>) -> tensor<4x9xbf16>
    %2 = stablehlo.multiply %0, %1 : tensor<4x9xbf16>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<2x9xbf16>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<2x9xbf16>
    %5 = stablehlo.multiply %3, %4 : tensor<2x9xbf16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %8 = stablehlo.multiply %6, %7 : tensor<1x9xbf16>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xbf16>) -> tensor<0x9xbf16>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<0x9xbf16>
    %11 = stablehlo.multiply %9, %10 : tensor<0x9xbf16>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xbf16>, tensor<0x9xbf16>) -> tensor<1x9xbf16>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xbf16>, tensor<bf16>) -> tensor<2x9xbf16>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xbf16>, tensor<bf16>) -> tensor<2x9xbf16>
    %18 = stablehlo.add %15, %17 : tensor<2x9xbf16>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<1x9xbf16>
    %21 = stablehlo.multiply %19, %20 : tensor<1x9xbf16>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<1x9xbf16>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xbf16>, tensor<1x9xbf16>) -> tensor<2x9xbf16>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xbf16>, tensor<bf16>) -> tensor<4x9xbf16>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xbf16>, tensor<bf16>) -> tensor<4x9xbf16>
    %28 = stablehlo.add %25, %27 : tensor<4x9xbf16>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<3x9xbf16>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xbf16>) -> tensor<3x9xbf16>
    %31 = stablehlo.multiply %29, %30 : tensor<3x9xbf16>
    %32 = "stablehlo.slice"(%arg0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xbf16>) -> tensor<1x9xbf16>
    %33 = stablehlo.concatenate %32, %31, dim = 0 : (tensor<1x9xbf16>, tensor<3x9xbf16>) -> tensor<4x9xbf16>
    %34 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %35 = stablehlo.pad %33, %34, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xbf16>, tensor<bf16>) -> tensor<8x9xbf16>
    %36 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %37 = stablehlo.pad %28, %36, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xbf16>, tensor<bf16>) -> tensor<8x9xbf16>
    %38 = stablehlo.add %35, %37 : tensor<8x9xbf16>
    return %38 : tensor<8x9xbf16>
  }
}
