// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.19.0' --verify-diagnostics --split-input-file %s

func.func @default_dynamic_conv(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>, %arg2: tensor<2x2xi32>) -> tensor<1x?x?x16xf32> {
  // expected-error @+2 {{failed to legalize operation 'vhlo.dynamic_conv_v2' that was explicitly marked illegal}}
  %d_padding = stablehlo.constant dense<0> : tensor<2x2xi32>
  %0 = "stablehlo.dynamic_conv"(%arg0, %arg1, %d_padding) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi32>) -> tensor<1x?x?x16xf32>
  func.return %0 : tensor<1x?x?x16xf32>
}
