// RUN: stablehlo-translate --interpret -split-input-file %s

module @cross_replica {
  func.func public @all_reduce(%operand : tensor<4xi64>) -> tensor<4xi64> {
    %result = "stablehlo.all_reduce"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<4xi64>) -> tensor<4xi64>
    return %result : tensor<4xi64>
  }
  func.func public @main() {
    %inputs0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %inputs1 = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[["all_reduce"], ["all_reduce"]]
    } : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    check.expect_eq_const %results#0, dense<[6, 8, 10, 12]> : tensor<4xi64>
    check.expect_eq_const %results#1, dense<[6, 8, 10, 12]> : tensor<4xi64>
    func.return
  }
}

// -----

module @cross_replica_and_partition {
  func.func public @all_reduce(%operand : tensor<4xi64>) -> tensor<4xi64> {
    %result = "stablehlo.all_reduce"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<4xi64>) -> tensor<4xi64>
    return %result : tensor<4xi64>
  }
  func.func public @main() {
    %inputs0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %inputs1 = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[["all_reduce"], ["all_reduce"]]
    } : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    check.expect_eq_const %results#0, dense<[6, 8, 10, 12]> : tensor<4xi64>
    check.expect_eq_const %results#1, dense<[6, 8, 10, 12]> : tensor<4xi64>
    func.return
  }
}

// -----

module @flattened_ids {
  func.func public @all_reduce(%operand : tensor<4xi64>) -> tensor<4xi64> {
    %result = "stablehlo.all_reduce"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids
    } : (tensor<4xi64>) -> tensor<4xi64>
    return %result : tensor<4xi64>
  }
  func.func public @main() {
    %inputs0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %inputs1 = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[["all_reduce"], ["all_reduce"]]
    } : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    check.expect_eq_const %results#0, dense<[6, 8, 10, 12]> : tensor<4xi64>
    check.expect_eq_const %results#1, dense<[6, 8, 10, 12]> : tensor<4xi64>
    func.return
  }
}

// -----

module @ragged_replica_groups {
  func.func public @all_reduce(%operand : tensor<4xi64>) -> tensor<4xi64> {
    %result = "stablehlo.all_reduce"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      replica_groups = dense<[[0, 1], [2, -1]]> : tensor<2x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<4xi64>) -> tensor<4xi64>
    return %result : tensor<4xi64>
  }
  func.func public @main() {
    %inputs0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %inputs1 = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
    %inputs2 = stablehlo.constant dense<[6, 8, 10, 12]> : tensor<4xi64>
    %results:3 = "interpreter.run_parallel"(%inputs0, %inputs1, %inputs2) {
      programs=[["all_reduce"], ["all_reduce"], ["all_reduce"]]
    } : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) ->
        (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)
    check.expect_eq_const %results#0, dense<[6, 8, 10, 12]> : tensor<4xi64>
    check.expect_eq_const %results#1, dense<[6, 8, 10, 12]> : tensor<4xi64>
    check.expect_eq_const %results#2, dense<[6, 8, 10, 12]> : tensor<4xi64>
    func.return
  }
}
