// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @dynamic_conv_op_test_si64() {
  %lhs = stablehlo.constant dense<[[
                                    [[1], [2], [5], [6]],
                                    [[3], [4], [7], [8]],
                                    [[10], [11], [14], [15]],
                                    [[12], [13], [16], [17]]
                                  ]]> : tensor<1x4x4x1xi64>
  %rhs = stablehlo.constant dense<1> : tensor<3x3x1x1xi64>
  %padding = stablehlo.constant dense<1> : tensor<2x2xi64>
  %result = "stablehlo.dynamic_conv"(%lhs, %rhs, %padding) {
    window_strides = array<i64: 4, 4>,
    lhs_dilation = array<i64: 2, 2>,
    rhs_dilation = array<i64: 1, 1>,
    window_reversal = array<i1: false, false>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>, tensor<2x2xi64>) -> tensor<1x2x2x1xi64>
  check.expect_eq_const %result, dense<[[
                                         [[1], [5]],
                                         [[10], [14]]
                                        ]]> : tensor<1x2x2x1xi64>
  func.return
}
