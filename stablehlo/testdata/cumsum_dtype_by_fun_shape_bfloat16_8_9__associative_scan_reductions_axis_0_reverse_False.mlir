// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xbf16>
    %1 = call @expected() : () -> tensor<8x9xbf16>
    %2 = call @cumsum(%0) : (tensor<8x9xbf16>) -> tensor<8x9xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xbf16>, tensor<8x9xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xbf16> {
    %0 = stablehlo.constant dense<[[3.640630e+00, 2.484380e+00, -1.601560e+00, -4.031250e+00, -4.843750e+00, 1.617190e+00, -4.500000e+00, -4.316410e-01, -4.199220e-02], [-5.718750e+00, 4.781250e+00, 5.718750e+00, -6.757810e-01, -1.049800e-01, -2.781250e+00, 3.515630e+00, -5.062500e+00, 3.750000e+00], [1.234380e+00, -1.039060e+00, 4.156250e+00, -2.031250e+00, 4.562500e+00, -2.671880e+00, -6.093750e-01, -3.796880e+00, -1.147460e-01], [-2.875000e+00, 1.826170e-01, -6.796880e-01, -1.796880e+00, -3.593750e-01, 2.843750e+00, -4.187500e+00, -5.843750e+00, 8.945310e-01], [5.312500e+00, -3.500000e+00, -1.109380e+00, 1.117190e+00, 2.062500e+00, -7.226560e-01, -1.492190e+00, 5.195310e-01, -1.320310e+00], [-1.976560e+00, -8.710930e-01, -8.984370e-01, 1.335940e+00, -1.007810e+00, 4.093750e+00, 1.921880e+00, 2.937500e+00, -1.187500e+00], [2.015630e+00, 3.765630e+00, -1.046880e+00, -1.703130e+00, 2.265630e+00, -7.460930e-01, 2.578130e-01, 1.218750e+00, -6.328130e-01], [-4.062500e+00, 4.187500e+00, -4.156250e+00, -4.343750e+00, 2.125000e+00, 4.437500e+00, 8.789060e-01, 1.000000e+00, -1.445310e+00]]> : tensor<8x9xbf16>
    return %0 : tensor<8x9xbf16>
  }
  func.func private @expected() -> tensor<8x9xbf16> {
    %0 = stablehlo.constant dense<[[3.640630e+00, 2.484380e+00, -1.601560e+00, -4.031250e+00, -4.843750e+00, 1.617190e+00, -4.500000e+00, -4.316410e-01, -4.199220e-02], [-2.078130e+00, 7.250000e+00, 4.125000e+00, -4.718750e+00, -4.937500e+00, -1.164060e+00, -9.843750e-01, -5.500000e+00, 3.703130e+00], [-8.437500e-01, 6.218750e+00, 8.250000e+00, -6.750000e+00, -3.750000e-01, -3.843750e+00, -1.593750e+00, -9.312500e+00, 3.593750e+00], [-3.718750e+00, 6.406250e+00, 7.593750e+00, -8.562500e+00, -7.500000e-01, -9.921870e-01, -5.812500e+00, -1.512500e+01, 4.500000e+00], [1.593750e+00, 2.906250e+00, 6.500000e+00, -7.437500e+00, 1.312500e+00, -1.718750e+00, -7.312500e+00, -1.462500e+01, 3.187500e+00], [-3.750000e-01, 2.031250e+00, 5.593750e+00, -6.125000e+00, 3.046880e-01, 2.375000e+00, -5.375000e+00, -1.168750e+01, 2.000000e+00], [1.640630e+00, 5.812500e+00, 4.562500e+00, -7.812500e+00, 2.562500e+00, 1.625000e+00, -5.125000e+00, -1.050000e+01, 1.367190e+00], [-2.421880e+00, 1.000000e+01, 4.062500e-01, -1.218750e+01, 4.687500e+00, 6.062500e+00, -4.250000e+00, -9.437500e+00, -6.250000e-02]]> : tensor<8x9xbf16>
    return %0 : tensor<8x9xbf16>
  }
  func.func private @cumsum(%arg0: tensor<8x9xbf16>) -> tensor<8x9xbf16> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xbf16>) -> tensor<4x9xbf16>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xbf16>) -> tensor<4x9xbf16>
    %2 = stablehlo.add %0, %1 : tensor<4x9xbf16>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<2x9xbf16>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<2x9xbf16>
    %5 = stablehlo.add %3, %4 : tensor<2x9xbf16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %8 = stablehlo.add %6, %7 : tensor<1x9xbf16>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xbf16>) -> tensor<0x9xbf16>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<0x9xbf16>
    %11 = stablehlo.add %9, %10 : tensor<0x9xbf16>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xbf16>, tensor<0x9xbf16>) -> tensor<1x9xbf16>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xbf16>, tensor<bf16>) -> tensor<2x9xbf16>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xbf16>, tensor<bf16>) -> tensor<2x9xbf16>
    %18 = stablehlo.add %15, %17 : tensor<2x9xbf16>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<1x9xbf16>
    %21 = stablehlo.add %19, %20 : tensor<1x9xbf16>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<1x9xbf16>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xbf16>, tensor<1x9xbf16>) -> tensor<2x9xbf16>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xbf16>, tensor<bf16>) -> tensor<4x9xbf16>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xbf16>, tensor<bf16>) -> tensor<4x9xbf16>
    %28 = stablehlo.add %25, %27 : tensor<4x9xbf16>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<3x9xbf16>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xbf16>) -> tensor<3x9xbf16>
    %31 = stablehlo.add %29, %30 : tensor<3x9xbf16>
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
