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
    %0 = stablehlo.constant dense<[[2.32396984, -3.15066099, -2.71259046, 1.12584889, -2.227380e+00, 1.34180832, -1.20343506, 5.611440e-01, 0.840916454], [-2.73314834, -1.94009256, -2.04174924, 0.620082437, -8.340550e+00, 0.888267576, -1.78298914, -4.30248594, 1.2075268], [3.86515188, -2.19563913, -2.33482265, 3.99685287, 1.77088594, 1.65483856, -0.296206266, 1.19495928, -2.33050179], [1.10100162, 2.44795752, 2.26104188, 2.02901721, -3.75156093, -1.04195237, 4.78353643, 1.64816082, 5.75762033], [-1.21566856, -0.406741798, -2.02965856, -2.06942964, -0.447444528, 1.64682186, -0.922979772, 4.03250265, 3.02044916], [-3.20794249, -4.31424856, -2.00080514, -4.18833399, -0.560978532, 5.23856974, -1.32336771, -2.90669036, -0.55723995], [-3.3236773, -3.62144017, 4.87731028, -0.331551135, 2.85932207, 1.53388393, 0.539640248, 4.43539619, -1.9167825], [0.272509247, -0.388932019, 5.90908957, -0.249131203, 1.22370291, 2.30888104, 3.7323246, -1.85150862, 0.356485814]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-3.3236773, -4.31424856, -2.71259046, -4.18833399, -8.340550e+00, -1.04195237, -1.78298914, -4.30248594, -2.33050179], [-3.3236773, -4.31424856, -2.33482265, -4.18833399, -8.340550e+00, -1.04195237, -1.78298914, -4.30248594, -2.33050179], [-3.3236773, -4.31424856, -2.33482265, -4.18833399, -3.75156093, -1.04195237, -1.32336771, -2.90669036, -2.33050179], [-3.3236773, -4.31424856, -2.02965856, -4.18833399, -3.75156093, -1.04195237, -1.32336771, -2.90669036, -1.9167825], [-3.3236773, -4.31424856, -2.02965856, -4.18833399, -0.560978532, 1.53388393, -1.32336771, -2.90669036, -1.9167825], [-3.3236773, -4.31424856, -2.00080514, -4.18833399, -0.560978532, 1.53388393, -1.32336771, -2.90669036, -1.9167825], [-3.3236773, -3.62144017, 4.87731028, -0.331551135, 1.22370291, 1.53388393, 0.539640248, -1.85150862, -1.9167825], [0.272509247, -0.388932019, 5.90908957, -0.249131203, 1.22370291, 2.30888104, 3.7323246, -1.85150862, 0.356485814]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cummin(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = stablehlo.reverse %arg0, dims = [0] : tensor<8x9xf32>
    %1 = "stablehlo.slice"(%0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %3 = stablehlo.minimum %1, %2 : tensor<4x9xf32>
    %4 = "stablehlo.slice"(%3) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %5 = "stablehlo.slice"(%3) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %6 = stablehlo.minimum %4, %5 : tensor<2x9xf32>
    %7 = "stablehlo.slice"(%6) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %8 = "stablehlo.slice"(%6) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %9 = stablehlo.minimum %7, %8 : tensor<1x9xf32>
    %10 = "stablehlo.slice"(%9) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf32>) -> tensor<0x9xf32>
    %11 = "stablehlo.slice"(%6) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<0x9xf32>
    %12 = stablehlo.minimum %10, %11 : tensor<0x9xf32>
    %13 = "stablehlo.slice"(%6) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %14 = stablehlo.concatenate %13, %12, dim = 0 : (tensor<1x9xf32>, tensor<0x9xf32>) -> tensor<1x9xf32>
    %15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.pad %14, %15, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.pad %9, %17, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %19 = stablehlo.add %16, %18 : tensor<2x9xf32>
    %20 = "stablehlo.slice"(%19) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %21 = "stablehlo.slice"(%3) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %22 = stablehlo.minimum %20, %21 : tensor<1x9xf32>
    %23 = "stablehlo.slice"(%3) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %24 = stablehlo.concatenate %23, %22, dim = 0 : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<2x9xf32>
    %25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.pad %24, %25, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.pad %19, %27, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %29 = stablehlo.add %26, %28 : tensor<4x9xf32>
    %30 = "stablehlo.slice"(%29) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<3x9xf32>
    %31 = "stablehlo.slice"(%0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<3x9xf32>
    %32 = stablehlo.minimum %30, %31 : tensor<3x9xf32>
    %33 = "stablehlo.slice"(%0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<1x9xf32>
    %34 = stablehlo.concatenate %33, %32, dim = 0 : (tensor<1x9xf32>, tensor<3x9xf32>) -> tensor<4x9xf32>
    %35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = stablehlo.pad %34, %35, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %38 = stablehlo.pad %29, %37, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %39 = stablehlo.add %36, %38 : tensor<8x9xf32>
    %40 = stablehlo.reverse %39, dims = [0] : tensor<8x9xf32>
    return %40 : tensor<8x9xf32>
  }
}
