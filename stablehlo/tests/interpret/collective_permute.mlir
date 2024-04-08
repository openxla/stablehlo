// RUN: stablehlo-translate --interpret -split-input-file %s

module @cross_replica {
  func.func @collective_permute(%operand : tensor<2x2xi64>) -> tensor<2x2xi64> {
    %result = "stablehlo.collective_permute"(%operand) {
      source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<2x2xi64>) -> tensor<2x2xi64>
    return %result : tensor<2x2xi64>
  }
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %inputs1 = stablehlo.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    %inputs2 = stablehlo.constant dense<[[9, 10], [11, 12]]> : tensor<2x2xi64>
    %results:3 = "interpreter.run_parallel"(%inputs0, %inputs1, %inputs2) {
      programs=[[@collective_permute], [@collective_permute], [@collective_permute]]
    } : (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>) ->
        (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>)
    check.expect_eq_const %results#0, dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>
    check.expect_eq_const %results#1, dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    check.expect_eq_const %results#2, dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    func.return
  }
}

// -----

module @cross_partition {
  func.func @collective_permute(%operand : tensor<2x2xi64>) -> tensor<2x2xi64> {
    %result = "stablehlo.collective_permute"(%operand) {
      source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<2x2xi64>) -> tensor<2x2xi64>
    return %result : tensor<2x2xi64>
  }
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %inputs1 = stablehlo.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    %inputs2 = stablehlo.constant dense<[[9, 10], [11, 12]]> : tensor<2x2xi64>
    %results:3 = "interpreter.run_parallel"(%inputs0, %inputs1, %inputs2) {
      programs=[[@collective_permute, @collective_permute, @collective_permute]]
    } : (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>) ->
        (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>)
    check.expect_eq_const %results#0, dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>
    check.expect_eq_const %results#1, dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    check.expect_eq_const %results#2, dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    func.return
  }
}
