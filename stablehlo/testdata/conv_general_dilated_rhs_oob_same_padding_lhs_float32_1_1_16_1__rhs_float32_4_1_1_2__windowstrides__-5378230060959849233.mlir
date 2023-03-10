// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x1x16x1xf32>, tensor<4x1x1x2xf32>)
    %1 = call @expected() : () -> tensor<1x1x16x2xf32>
    %2 = stablehlo.convolution(%0#0, %0#1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 2], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x16x1xf32>, tensor<4x1x1x2xf32>) -> tensor<1x1x16x2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1x1x16x2xf32>, tensor<1x1x16x2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x1x16x1xf32>, tensor<4x1x1x2xf32>) {
    %0 = stablehlo.constant dense<[[[[1.94757128], [0.731765032], [-0.726273656], [-5.65812302], [-0.322147697], [2.76183558], [4.334000e+00], [0.198797315], [0.552351534], [-3.67483807], [1.00321078], [1.13255763], [1.41218019], [2.263990e+00], [-3.61878276], [2.55396557]]]]> : tensor<1x1x16x1xf32>
    %1 = stablehlo.constant dense<[[[[-2.90031719, 3.83649588]]], [[[3.26177287, 6.146370e-01]]], [[[2.2365787, -2.09627247]]], [[[-2.90434122, -0.155085653]]]]> : tensor<4x1x1x2xf32>
    return %0, %1 : tensor<1x1x16x1xf32>, tensor<4x1x1x2xf32>
  }
  func.func private @expected() -> tensor<1x1x16x2xf32> {
    %0 = stablehlo.constant dense<[[[[6.35253525, 1.19704938], [2.38685131, 0.449769884], [-2.36893964, -0.446394682], [-18.455513, -3.47769189], [-1.05077267, -0.198003903], [9.008480e+00, 1.69752634], [14.1365242, 2.66383696], [0.648431718, 0.122188188], [1.80164528, 0.339495689], [-11.9864874, -2.25869155], [3.27224565, 0.616610467], [3.69414568, 0.696111857], [4.60621119, 0.867978215], [7.38462114, 1.39153206], [-11.803647, -2.22423792], [8.33045578, 1.56976175]]]]> : tensor<1x1x16x2xf32>
    return %0 : tensor<1x1x16x2xf32>
  }
}

