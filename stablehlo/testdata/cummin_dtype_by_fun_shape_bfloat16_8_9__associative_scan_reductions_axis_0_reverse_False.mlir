// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xbf16>
    %1 = call @expected() : () -> tensor<8x9xbf16>
    %2 = call @cummin(%0) : (tensor<8x9xbf16>) -> tensor<8x9xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xbf16>, tensor<8x9xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xbf16> {
    %0 = stablehlo.constant dense<[[4.718750e+00, 2.703130e+00, -2.468750e+00, -1.699220e-01, -8.242180e-01, 7.468750e+00, -3.968750e+00, -3.164060e-01, -7.226560e-01], [-1.718750e-01, 6.914060e-01, 8.593750e-01, 2.050780e-01, -1.734380e+00, -4.570310e-01, 2.625000e+00, 4.296880e-01, -2.171880e+00], [4.726560e-01, 5.875000e+00, 2.093750e+00, 1.046880e+00, 3.640630e+00, 3.531250e+00, 3.062500e+00, 1.164060e+00, -1.734380e+00], [2.451170e-01, 1.570310e+00, 2.656250e+00, -1.992190e+00, -2.218750e+00, 3.000000e+00, -5.742190e-01, 1.476560e+00, 3.312500e+00], [3.453130e+00, 2.796880e+00, -3.484380e+00, -3.156250e+00, 9.140620e-01, 6.156250e+00, 4.863280e-01, 1.843750e+00, -3.109380e+00], [-1.773440e+00, 9.726560e-01, 1.945310e+00, 4.562500e+00, 3.109380e+00, 2.859380e+00, 2.468750e+00, -3.671880e+00, 1.203130e+00], [4.000000e+00, -1.367190e+00, 1.281250e+00, 4.343750e+00, -3.828130e+00, -1.318360e-01, 1.257810e+00, 4.250000e+00, 1.898440e+00], [2.843750e+00, -3.173830e-02, -9.960930e-01, 5.968750e+00, 1.773440e+00, 3.234380e+00, -6.562500e+00, -6.718750e-01, 2.078130e+00]]> : tensor<8x9xbf16>
    return %0 : tensor<8x9xbf16>
  }
  func.func private @expected() -> tensor<8x9xbf16> {
    %0 = stablehlo.constant dense<[[4.718750e+00, 2.703130e+00, -2.468750e+00, -1.699220e-01, -8.242180e-01, 7.468750e+00, -3.968750e+00, -3.164060e-01, -7.226560e-01], [-1.718750e-01, 6.914060e-01, -2.468750e+00, -1.699220e-01, -1.734380e+00, -4.570310e-01, -3.968750e+00, -3.164060e-01, -2.171880e+00], [-1.718750e-01, 6.914060e-01, -2.468750e+00, -1.699220e-01, -1.734380e+00, -4.570310e-01, -3.968750e+00, -3.164060e-01, -2.171880e+00], [-1.718750e-01, 6.914060e-01, -2.468750e+00, -1.992190e+00, -2.218750e+00, -4.570310e-01, -3.968750e+00, -3.164060e-01, -2.171880e+00], [-1.718750e-01, 6.914060e-01, -3.484380e+00, -3.156250e+00, -2.218750e+00, -4.570310e-01, -3.968750e+00, -3.164060e-01, -3.109380e+00], [-1.773440e+00, 6.914060e-01, -3.484380e+00, -3.156250e+00, -2.218750e+00, -4.570310e-01, -3.968750e+00, -3.671880e+00, -3.109380e+00], [-1.773440e+00, -1.367190e+00, -3.484380e+00, -3.156250e+00, -3.828130e+00, -4.570310e-01, -3.968750e+00, -3.671880e+00, -3.109380e+00], [-1.773440e+00, -1.367190e+00, -3.484380e+00, -3.156250e+00, -3.828130e+00, -4.570310e-01, -6.562500e+00, -3.671880e+00, -3.109380e+00]]> : tensor<8x9xbf16>
    return %0 : tensor<8x9xbf16>
  }
  func.func private @cummin(%arg0: tensor<8x9xbf16>) -> tensor<8x9xbf16> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xbf16>) -> tensor<4x9xbf16>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xbf16>) -> tensor<4x9xbf16>
    %2 = stablehlo.minimum %0, %1 : tensor<4x9xbf16>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<2x9xbf16>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<2x9xbf16>
    %5 = stablehlo.minimum %3, %4 : tensor<2x9xbf16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %8 = stablehlo.minimum %6, %7 : tensor<1x9xbf16>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xbf16>) -> tensor<0x9xbf16>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<0x9xbf16>
    %11 = stablehlo.minimum %9, %10 : tensor<0x9xbf16>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xbf16>, tensor<0x9xbf16>) -> tensor<1x9xbf16>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xbf16>, tensor<bf16>) -> tensor<2x9xbf16>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xbf16>, tensor<bf16>) -> tensor<2x9xbf16>
    %18 = stablehlo.add %15, %17 : tensor<2x9xbf16>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xbf16>) -> tensor<1x9xbf16>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<1x9xbf16>
    %21 = stablehlo.minimum %19, %20 : tensor<1x9xbf16>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<1x9xbf16>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xbf16>, tensor<1x9xbf16>) -> tensor<2x9xbf16>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xbf16>, tensor<bf16>) -> tensor<4x9xbf16>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xbf16>, tensor<bf16>) -> tensor<4x9xbf16>
    %28 = stablehlo.add %25, %27 : tensor<4x9xbf16>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xbf16>) -> tensor<3x9xbf16>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xbf16>) -> tensor<3x9xbf16>
    %31 = stablehlo.minimum %29, %30 : tensor<3x9xbf16>
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
