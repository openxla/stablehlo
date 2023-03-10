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
    %0 = stablehlo.constant dense<[[4.50501823, -2.2358222, 2.18727183, 4.83398962, 1.98953748, -1.23601794, -1.83440244, -2.62308359, 1.75403392], [-1.23143733, 2.2988584, -2.60907102, 2.77088404, -3.33188558, -3.29199481, 5.44026136, -2.09491062, 0.0293852929], [6.14963865, -0.884192705, 3.19141078, 3.78954959, 3.1674552, -2.04460526, -1.2209419, -1.32764864, 1.86086059], [0.22357747, 2.66421771, -3.84891367, -2.23127866, -1.29162383, 0.90816456, -4.45007086, 3.05411696, -3.436190e+00], [-2.15938473, -5.89252853, 0.267026782, -0.966214179, -4.56652212, -0.452082962, -1.13843489, 0.464001268, 0.571039915], [-1.55870354, -1.79844129, 3.43749189, 2.78967857, 0.0503596514, 4.592100e+00, 0.823536872, -7.16196489, -2.67309356], [2.57470703, 1.48755395, -1.6614269, -1.61904049, 6.11397028, 0.749588191, -7.3956232, -2.97237062, 4.99997711], [4.515160e+00, -1.57953441, -3.38962579, -0.460906237, -1.87051785, 0.20963572, 6.29518461, -2.20884132, -2.53437614]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[4.50501823, -2.2358222, 2.18727183, 4.83398962, 1.98953748, -1.23601794, -1.83440244, -2.62308359, 1.75403392], [4.50501823, 2.2988584, 2.18727183, 4.83398962, 1.98953748, -1.23601794, 5.44026136, -2.09491062, 1.75403392], [6.14963865, 2.2988584, 3.19141078, 4.83398962, 3.1674552, -1.23601794, 5.44026136, -1.32764864, 1.86086059], [6.14963865, 2.66421771, 3.19141078, 4.83398962, 3.1674552, 0.90816456, 5.44026136, 3.05411696, 1.86086059], [6.14963865, 2.66421771, 3.19141078, 4.83398962, 3.1674552, 0.90816456, 5.44026136, 3.05411696, 1.86086059], [6.14963865, 2.66421771, 3.43749189, 4.83398962, 3.1674552, 4.592100e+00, 5.44026136, 3.05411696, 1.86086059], [6.14963865, 2.66421771, 3.43749189, 4.83398962, 6.11397028, 4.592100e+00, 5.44026136, 3.05411696, 4.99997711], [6.14963865, 2.66421771, 3.43749189, 4.83398962, 6.11397028, 4.592100e+00, 6.29518461, 3.05411696, 4.99997711]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cummax(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<4x9xf32>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %5 = stablehlo.maximum %3, %4 : tensor<2x9xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %8 = stablehlo.maximum %6, %7 : tensor<1x9xf32>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf32>) -> tensor<0x9xf32>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<0x9xf32>
    %11 = stablehlo.maximum %9, %10 : tensor<0x9xf32>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xf32>, tensor<0x9xf32>) -> tensor<1x9xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %18 = stablehlo.add %15, %17 : tensor<2x9xf32>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %21 = stablehlo.maximum %19, %20 : tensor<1x9xf32>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<2x9xf32>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %28 = stablehlo.add %25, %27 : tensor<4x9xf32>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<3x9xf32>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<3x9xf32>
    %31 = stablehlo.maximum %29, %30 : tensor<3x9xf32>
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
