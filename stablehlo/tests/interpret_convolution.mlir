// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @convolution() {
  %lhs = stablehlo.constant dense<[[
                                    [[1], [2], [5], [6]],
                                    [[3], [4], [7], [8]],
                                    [[10], [11], [14], [15]],
                                    [[12], [13], [16], [17]]
                                  ]]> : tensor<1x4x4x1xi64>
  %rhs = stablehlo.constant dense<[[[[1]],
                                    [[1]],
                                    [[1]]],
                                    [[[1]],
                                    [[1]],
                                    [[1]]],
                                    [[[1]],
                                    [[1]],
                                    [[1]]]]> : tensor<3x3x1x1xi64>
  %result = stablehlo.convolution(%lhs, %rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [4, 4],
      pad = [[0, 0], [0, 0]],
      lhs_dilate = [2, 2],
      rhs_dilate = [1, 1],
      reverse = [false, false]
    } {
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    }
  : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>) -> tensor<1x2x2x1xi64>
  check.expect_eq_const %result, dense<[[
                                          [[10], [26]],
                                          [[46], [62]]
                                        ]]> : tensor<1x2x2x1xi64>
  func.return
}
