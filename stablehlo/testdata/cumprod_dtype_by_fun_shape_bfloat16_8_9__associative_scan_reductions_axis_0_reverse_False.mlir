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
    %0 = stablehlo.constant dense<[[-3.390630e+00, 3.734380e+00, 3.906250e+00, 4.968750e+00, -2.375000e+00, -1.484380e+00, 1.109380e+00, -1.069340e-01, -3.500000e+00], [-5.234380e-01, 1.070310e+00, 2.281250e+00, -6.875000e-01, -1.757810e+00, -1.273440e+00, 1.945310e+00, -3.437500e+00, -4.500000e+00], [-2.812500e+00, -4.156250e+00, 5.593750e+00, 3.796880e+00, 2.359380e+00, -4.843750e+00, -2.343750e+00, 1.210940e+00, 2.171880e+00], [-2.984380e+00, 5.687500e+00, 4.609380e-01, 4.281250e+00, -2.578130e+00, 1.031250e+00, 4.726560e-01, 1.789060e+00, -1.796880e+00], [-2.949220e-01, 5.781250e+00, 7.031250e-01, -7.968750e-01, -2.500000e+00, 2.437500e+00, -1.376950e-01, 3.437500e-01, -1.359380e+00], [-1.953130e+00, 6.757810e-01, -4.562500e+00, -3.156250e+00, -2.906250e+00, 2.531250e+00, 6.562500e+00, 3.156250e+00, -3.250000e+00], [-2.812500e+00, -1.242190e+00, 1.085940e+00, -2.375000e+00, 1.640630e+00, 1.601560e+00, -1.867190e+00, 1.203130e+00, -2.765630e+00], [-1.250000e+00, 3.187500e+00, -9.619140e-02, -3.843750e+00, -1.031250e+00, 1.078130e+00, 6.367190e-01, 1.328130e+00, 3.281250e+00]]> : tensor<8x9xbf16>
    return %0 : tensor<8x9xbf16>
  }
  func.func private @expected() -> tensor<8x9xbf16> {
    %0 = stablehlo.constant dense<[[-3.390630e+00, 3.734380e+00, 3.906250e+00, 4.968750e+00, -2.375000e+00, -1.484380e+00, 1.109380e+00, -1.069340e-01, -3.500000e+00], [1.773440e+00, 4.000000e+00, 8.937500e+00, -3.421880e+00, 4.187500e+00, 1.890630e+00, 2.156250e+00, 3.671880e-01, 1.575000e+01], [-5.000000e+00, -1.662500e+01, 5.000000e+01, -1.300000e+01, 9.875000e+00, -9.187500e+00, -5.062500e+00, 4.453130e-01, 3.425000e+01], [1.487500e+01, -9.450000e+01, 2.300000e+01, -5.550000e+01, -2.550000e+01, -9.437500e+00, -2.390630e+00, 7.968750e-01, -6.150000e+01], [-4.375000e+00, -5.480000e+02, 1.612500e+01, 4.425000e+01, 6.375000e+01, -2.300000e+01, 3.300780e-01, 2.734380e-01, 8.350000e+01], [8.562500e+00, -3.700000e+02, -7.350000e+01, -1.400000e+02, -1.850000e+02, -5.800000e+01, 2.156250e+00, 8.671870e-01, -2.700000e+02], [-2.412500e+01, 4.600000e+02, -8.000000e+01, 3.320000e+02, -3.040000e+02, -9.300000e+01, -4.031250e+00, 1.046880e+00, 7.480000e+02], [3.000000e+01, 1.456000e+03, 7.687500e+00, -1.280000e+03, 3.140000e+02, -1.005000e+02, -2.562500e+00, 1.390630e+00, 2.464000e+03]]> : tensor<8x9xbf16>
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
