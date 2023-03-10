// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf32>
    %1 = call @expected() : () -> tensor<8x9xf32>
    %2 = call @cummin(%0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf32>, tensor<8x9xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[4.53644323, 2.1770215, -7.00054264, -2.60454726, 5.51019526, 0.333813608, -4.23384857, 0.212504104, 3.17025566], [1.88752055, 4.06542492, 0.801184058, -2.48842931, 4.52777863, -0.465400845, -1.09660077, -1.53955877, -3.01447916], [0.620056569, -4.099970e-01, -1.53796887, -1.46075273, 0.0836434587, 3.17984748, -2.74627209, -4.394360e+00, 1.69851148], [0.414670259, -3.41243935, 0.370075673, 0.264903814, 3.08682513, -5.94163752, -1.84098125, -4.57074308, 3.98163867], [-0.850515842, -3.3095479, -3.64026737, -4.02322245, 0.376517922, -0.270804495, 2.61304283, -3.18589711, 1.69072533], [5.854080e-01, -3.79898095, -2.98369098, 1.08989036, -3.1441431, -9.53399467, -0.3063173, 1.49476051, -3.2479651], [1.44614577, -0.0067521669, -8.94898223, -4.12914181, -4.73207617, 5.15799284, -3.33543301, -3.73492074, 4.58027029], [-5.39144516, 2.11447573, -1.89683664, -0.394717246, 2.13657403, 1.20620215, 3.32112479, -4.36102724, 0.187602013]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[4.53644323, 2.1770215, -7.00054264, -2.60454726, 5.51019526, 0.333813608, -4.23384857, 0.212504104, 3.17025566], [1.88752055, 2.1770215, -7.00054264, -2.60454726, 4.52777863, -0.465400845, -4.23384857, -1.53955877, -3.01447916], [0.620056569, -4.099970e-01, -7.00054264, -2.60454726, 0.0836434587, -0.465400845, -4.23384857, -4.394360e+00, -3.01447916], [0.414670259, -3.41243935, -7.00054264, -2.60454726, 0.0836434587, -5.94163752, -4.23384857, -4.57074308, -3.01447916], [-0.850515842, -3.41243935, -7.00054264, -4.02322245, 0.0836434587, -5.94163752, -4.23384857, -4.57074308, -3.01447916], [-0.850515842, -3.79898095, -7.00054264, -4.02322245, -3.1441431, -9.53399467, -4.23384857, -4.57074308, -3.2479651], [-0.850515842, -3.79898095, -8.94898223, -4.12914181, -4.73207617, -9.53399467, -4.23384857, -4.57074308, -3.2479651], [-5.39144516, -3.79898095, -8.94898223, -4.12914181, -4.73207617, -9.53399467, -4.23384857, -4.57074308, -3.2479651]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cummin(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %2 = stablehlo.minimum %0, %1 : tensor<4x9xf32>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %5 = stablehlo.minimum %3, %4 : tensor<2x9xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %8 = stablehlo.minimum %6, %7 : tensor<1x9xf32>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf32>) -> tensor<0x9xf32>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<0x9xf32>
    %11 = stablehlo.minimum %9, %10 : tensor<0x9xf32>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xf32>, tensor<0x9xf32>) -> tensor<1x9xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %18 = stablehlo.add %15, %17 : tensor<2x9xf32>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %21 = stablehlo.minimum %19, %20 : tensor<1x9xf32>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<2x9xf32>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %28 = stablehlo.add %25, %27 : tensor<4x9xf32>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<3x9xf32>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<3x9xf32>
    %31 = stablehlo.minimum %29, %30 : tensor<3x9xf32>
    %32 = "stablehlo.slice"(%arg0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<1x9xf32>
    %33 = stablehlo.concatenate %32, %31, dim = 0 : (tensor<1x9xf32>, tensor<3x9xf32>) -> tensor<4x9xf32>
    %34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %35 = stablehlo.pad %33, %34, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %37 = stablehlo.pad %28, %36, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %38 = stablehlo.add %35, %37 : tensor<8x9xf32>
    return %38 : tensor<8x9xf32>
  }
}
