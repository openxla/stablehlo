// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @convolution_op_test_si64() {
  %lhs = stablehlo.constant dense<[[
                                    [[1], [2], [5], [6]],
                                    [[3], [4], [7], [8]],
                                    [[10], [11], [14], [15]],
                                    [[12], [13], [16], [17]]
                                  ]]> : tensor<1x4x4x1xi64>
  %rhs = stablehlo.constant dense<1> : tensor<3x3x1x1xi64>
  %result = stablehlo.convolution(%lhs, %rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [4, 4],
      lhs_dilate = [2, 2]
    } {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    }
  : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>) -> tensor<1x2x2x1xi64>
  check.expect_eq_const %result, dense<[[
                                         [[10], [26]],
                                         [[46], [62]]
                                        ]]> : tensor<1x2x2x1xi64>
  func.return
}

// -----

func.func @convolution_op_test_padding() {
  %lhs = stablehlo.constant dense<[[
                                    [[1], [2], [5], [6]],
                                    [[3], [4], [7], [8]],
                                    [[10], [11], [14], [15]],
                                    [[12], [13], [16], [17]]
                                  ]]> : tensor<1x4x4x1xi64>
  %rhs = stablehlo.constant dense<1> : tensor<3x3x1x1xi64>
  %result = stablehlo.convolution(%lhs, %rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [4, 4],
      pad = [[1, 1], [1, 1]],
      lhs_dilate = [2, 2]
    } {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    }
  : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>) -> tensor<1x2x2x1xi64>
  check.expect_eq_const %result, dense<[[
                                         [[1], [5]],
                                         [[10], [14]]
                                        ]]> : tensor<1x2x2x1xi64>
  func.return
}

// -----

func.func @convolution_batch_group_count_4() {
  %lhs = stablehlo.constant dense<[[
                                    [[1], [2], [5], [6]],
                                    [[3], [4], [7], [8]],
                                    [[10], [11], [14], [15]],
                                    [[12], [13], [16], [17]]
                                  ]]> : tensor<1x4x4x1xi64>
  %rhs = stablehlo.constant dense<1> : tensor<1x2x1x4xi64>
  %result = stablehlo.convolution(%lhs, %rhs)
    dim_numbers = [0, b, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [4, 4],
      lhs_dilate = [2, 2]
    } {
      batch_group_count = 4 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    }
  : (tensor<1x4x4x1xi64>, tensor<1x2x1x4xi64>) -> tensor<1x1x2x4xi64>
  check.expect_eq_const %result, dense<[[[[1, 3, 10, 12],
                                          [5, 7, 14, 16]]]]> : tensor<1x1x2x4xi64>
  func.return
}

// -----

func.func @convolution_feature_group_count_2() {
  %lhs = stablehlo.constant dense<[[
                                    [[1], [2], [5], [6]],
                                    [[3], [4], [7], [8]],
                                    [[10], [11], [14], [15]],
                                    [[12], [13], [16], [17]]
                                  ]]> : tensor<1x4x4x1xi64>
  %rhs = stablehlo.constant dense<1> : tensor<1x2x1x4xi64>
  %result = stablehlo.convolution(%lhs, %rhs)
    dim_numbers = [b, 0, f, 1]x[0, i, 1, o]->[b, 0, 1, f],
    window = {
      stride = [4, 4],
      lhs_dilate = [2, 2]
    } {
      batch_group_count = 1 : i64,
      feature_group_count = 2 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    }
  : (tensor<1x4x4x1xi64>, tensor<1x2x1x4xi64>) -> tensor<1x2x1x4xi64>
  check.expect_eq_const %result, dense<[[[[3, 3, 11, 11]],
                                         [[21, 21, 29, 29]]]]> : tensor<1x2x1x4xi64>
  func.return
}
