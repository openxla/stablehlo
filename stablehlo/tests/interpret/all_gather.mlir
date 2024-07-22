// RUN: stablehlo-translate --interpret -split-input-file %s

module @cross_replica {
  func.func @all_gather(%arg0 : tensor<2x2xi64>) -> tensor<2x4xi64> {
    %result = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x2xi64>) -> tensor<2x4xi64>
    return %result : tensor<2x4xi64>
  }
  func.func @main() {
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
  func.func @all_gather(%arg0 : tensor<2x2xi64>) -> tensor<2x4xi64> {
    %result = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<2x2xi64>) -> tensor<2x4xi64>
    return %result : tensor<2x4xi64>
  }
  func.func @main() {
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

module @cross_replica_and_partition_issue_1933 {
  func.func @all_gather(%arg0 : tensor<2x2xi64>) -> tensor<2x8xi64> {
    %result = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle=1, type=0>
    } : (tensor<2x2xi64>) -> tensor<2x8xi64>
    return %result : tensor<2x8xi64>
  }
  func.func @main() {
    %0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %1 = stablehlo.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    %results:4 = "interpreter.run_parallel"(%1, %1, %0, %1) {
      programs=[[@all_gather, @all_gather], [@all_gather, @all_gather]]
    } : (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>) ->
        (tensor<2x8xi64>, tensor<2x8xi64>, tensor<2x8xi64>, tensor<2x8xi64>)
    check.expect_eq_const %results#0, dense<[[5, 6, 1, 2, 5, 6, 5, 6],
                                             [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>
    check.expect_eq_const %results#1, dense<[[5, 6, 1, 2, 5, 6, 5, 6],
                                             [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>
    check.expect_eq_const %results#2, dense<[[5, 6, 1, 2, 5, 6, 5, 6],
                                             [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>
    check.expect_eq_const %results#3, dense<[[5, 6, 1, 2, 5, 6, 5, 6],
                                             [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>

    func.return
  }
}

// -----

module @flattened_ids {
  func.func @all_gather(%arg0 : tensor<2x2xi64>) -> tensor<2x4xi64> {
    %result = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids
    } : (tensor<2x2xi64>) -> tensor<2x4xi64>
    return %result : tensor<2x4xi64>
  }
  func.func @main() {
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

module @cross_replica_variadic_inputs {
  func.func @all_gather(%arg0 : tensor<2x2xi64>, %arg1 : tensor<2x2xi32>) -> (tensor<2x4xi64>, tensor<2x4xi32>) {
    %result:2 = "stablehlo.all_gather"(%arg0, %arg1) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x2xi64>, tensor<2x2xi32>) -> (tensor<2x4xi64>, tensor<2x4xi32>)
    return %result#0, %result#1 : tensor<2x4xi64>, tensor<2x4xi32>
  }
  func.func @main() {
    %process0_operand0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %process0_operand1 = stablehlo.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi32>
    %process1_operand0 = stablehlo.constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi64>
    %process1_operand1 = stablehlo.constant dense<[[15, 16], [17, 18]]> : tensor<2x2xi32>
    %results:4 = "interpreter.run_parallel"(%process0_operand0, %process0_operand1, %process1_operand0, %process1_operand1) {
      programs=[[@all_gather], [@all_gather]]
    } : (tensor<2x2xi64>, tensor<2x2xi32>, tensor<2x2xi64>, tensor<2x2xi32>) -> (tensor<2x4xi64>, tensor<2x4xi32>, tensor<2x4xi64>, tensor<2x4xi32>)
    check.expect_eq_const %results#0, dense<[[1, 2, 11, 12],
                                             [3, 4, 13, 14]]> : tensor<2x4xi64>
    check.expect_eq_const %results#1, dense<[[5, 6, 15, 16],
                                             [7, 8, 17, 18]]> : tensor<2x4xi32>
    check.expect_eq_const %results#2, dense<[[1, 2, 11, 12],
                                             [3, 4, 13, 14]]> : tensor<2x4xi64>
    check.expect_eq_const %results#3, dense<[[5, 6, 15, 16],
                                             [7, 8, 17, 18]]> : tensor<2x4xi32>
    func.return
  }
}
