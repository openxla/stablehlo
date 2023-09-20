// RUN: stablehlo-translate --interpret -split-input-file %s

module @cross_replica {
  func.func public @all_gather(%arg0 : tensor<2x2xi64>) -> tensor<2x4xi64> {
    %result = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x2xi64>) -> tensor<2x4xi64>
    return %result : tensor<2x4xi64>
  }
  func.func public @main() {
    %0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %1 = stablehlo.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    %results:2 = "interpreter.run_parallel"(%0, %1) {
      programs=[[@all_gather], [@all_gather]]
    } : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
    check.expect_eq_const %results#0, dense<[[1, 2, 5, 6],
                                             [3, 4, 7, 8]]> : tensor<2x4xi64>
    check.expect_eq_const %results#1, dense<[[1, 2, 5, 6],
                                             [3, 4, 7, 8]]> : tensor<2x4xi64>
    func.return
  }
}

// -----

module @cross_replica_and_partition {
  func.func public @all_gather(%arg0 : tensor<2x2xi64>) -> tensor<2x4xi64> {
    %result = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<2x2xi64>) -> tensor<2x4xi64>
    return %result : tensor<2x4xi64>
  }
  func.func public @main() {
    %0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %1 = stablehlo.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    %results:2 = "interpreter.run_parallel"(%0, %1) {
      programs=[[@all_gather], [@all_gather]]
    } : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
    check.expect_eq_const %results#0, dense<[[1, 2, 5, 6],
                                             [3, 4, 7, 8]]> : tensor<2x4xi64>
    check.expect_eq_const %results#1, dense<[[1, 2, 5, 6],
                                             [3, 4, 7, 8]]> : tensor<2x4xi64>
    func.return
  }
}

// -----

module @flattened_ids {
  func.func public @all_gather(%arg0 : tensor<2x2xi64>) -> tensor<2x4xi64> {
    %result = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids
    } : (tensor<2x2xi64>) -> tensor<2x4xi64>
    return %result : tensor<2x4xi64>
  }
  func.func public @main() {
    %0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %1 = stablehlo.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    %results:2 = "interpreter.run_parallel"(%0, %1) {
      programs=[[@all_gather], [@all_gather]]
    } : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
    check.expect_eq_const %results#0, dense<[[1, 2, 5, 6],
                                             [3, 4, 7, 8]]> : tensor<2x4xi64>
    check.expect_eq_const %results#1, dense<[[1, 2, 5, 6],
                                             [3, 4, 7, 8]]> : tensor<2x4xi64>
    func.return
  }
}
