// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf32>
    %1 = call @expected() : () -> tensor<8x9xf32>
    %2 = call @cummax(%0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf32>, tensor<8x9xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-0.100494184, 1.93613255, -0.813259422, 0.81960982, 1.41883075, -0.165638193, 0.748748064, 1.49570584, -3.48141694], [-0.392679542, -3.2818296, -3.37848091, 1.27059197, -3.05341172, -0.376013815, 0.538633645, 4.51036406, -0.724414348], [-1.66856217, -5.28973627, -1.13861847, -0.0305497162, -0.348608226, -0.622938871, -3.75504827, -0.700232506, -0.971391558], [4.67770624, -1.68206382, -4.44312429, 0.377149165, 1.21928847, 0.185133293, 0.937096297, 2.68182302, -2.74890733], [1.613042, 7.15669775, 1.31272089, 0.165312767, 2.2985816, -3.72340345, -2.41147757, 1.71071613, -2.38085675], [3.34720707, -3.70331812, -2.19794655, 2.49207544, 2.42819142, 0.59212178, -3.94590831, 0.895995318, 3.93645191], [-5.60201836, -2.26534653, 1.42376041, 3.92853475, 1.16189134, 3.35934329, -3.7594738, 4.47656441, -3.25689673], [-0.0497927032, -3.14663458, -5.651050e+00, -6.87600374, -1.47560167, -0.83394891, -1.0004375, -0.869617879, -2.50005078]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-0.100494184, 1.93613255, 1.93613255, 1.93613255, 1.93613255, 1.93613255, 1.93613255, 1.93613255, 1.93613255], [-0.392679542, -0.392679542, -0.392679542, 1.27059197, 1.27059197, 1.27059197, 1.27059197, 4.51036406, 4.51036406], [-1.66856217, -1.66856217, -1.13861847, -0.0305497162, -0.0305497162, -0.0305497162, -0.0305497162, -0.0305497162, -0.0305497162], [4.67770624, 4.67770624, 4.67770624, 4.67770624, 4.67770624, 4.67770624, 4.67770624, 4.67770624, 4.67770624], [1.613042, 7.15669775, 7.15669775, 7.15669775, 7.15669775, 7.15669775, 7.15669775, 7.15669775, 7.15669775], [3.34720707, 3.34720707, 3.34720707, 3.34720707, 3.34720707, 3.34720707, 3.34720707, 3.34720707, 3.93645191], [-5.60201836, -2.26534653, 1.42376041, 3.92853475, 3.92853475, 3.92853475, 3.92853475, 4.47656441, 4.47656441], [-0.0497927032, -0.0497927032, -0.0497927032, -0.0497927032, -0.0497927032, -0.0497927032, -0.0497927032, -0.0497927032, -0.0497927032]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cummax(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<8> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<8x4xf32>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 3]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x2xf32>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 4]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x2xf32>
    %5 = stablehlo.maximum %3, %4 : tensor<8x2xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %8 = stablehlo.maximum %6, %7 : tensor<8x1xf32>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[8, 0]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x1xf32>) -> tensor<8x0xf32>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 2]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x0xf32>
    %11 = stablehlo.maximum %9, %10 : tensor<8x0xf32>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %13 = stablehlo.concatenate %12, %11, dim = 1 : (tensor<8x1xf32>, tensor<8x0xf32>) -> tensor<8x1xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<8x1xf32>, tensor<f32>) -> tensor<8x2xf32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.pad %8, %16, low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<8x1xf32>, tensor<f32>) -> tensor<8x2xf32>
    %18 = stablehlo.add %15, %17 : tensor<8x2xf32>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 4]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x1xf32>
    %21 = stablehlo.maximum %19, %20 : tensor<8x1xf32>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x1xf32>
    %23 = stablehlo.concatenate %22, %21, dim = 1 : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x2xf32>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8x4xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.pad %18, %26, low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8x4xf32>
    %28 = stablehlo.add %25, %27 : tensor<8x4xf32>
    %29 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %30 = stablehlo.maximum %28, %29 : tensor<8x4xf32>
    %31 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x1xf32>
    %32 = stablehlo.concatenate %31, %30, dim = 1 : (tensor<8x1xf32>, tensor<8x4xf32>) -> tensor<8x5xf32>
    %33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.pad %32, %33, low = [0, 0], high = [0, 0], interior = [0, 1] : (tensor<8x5xf32>, tensor<f32>) -> tensor<8x9xf32>
    %35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = stablehlo.pad %28, %35, low = [0, 1], high = [0, 1], interior = [0, 1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8x9xf32>
    %37 = stablehlo.add %34, %36 : tensor<8x9xf32>
    return %37 : tensor<8x9xf32>
  }
}
