// RUN: stablehlo-translate --interpret -split-input-file %s

module @cross_replica {
  func.func @all_reduce(%operand : tensor<4xi64>) -> tensor<4xi64> {
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
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %inputs1 = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@all_reduce], [@all_reduce]]
    } : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    check.expect_eq_const %results#0, dense<[6, 8, 10, 12]> : tensor<4xi64>
    check.expect_eq_const %results#1, dense<[6, 8, 10, 12]> : tensor<4xi64>
    func.return
  }
}

// -----

module @cross_replica_and_partition {
  func.func @all_reduce(%operand : tensor<4xi64>) -> tensor<4xi64> {
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
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %inputs1 = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@all_reduce], [@all_reduce]]
    } : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    check.expect_eq_const %results#0, dense<[6, 8, 10, 12]> : tensor<4xi64>
    check.expect_eq_const %results#1, dense<[6, 8, 10, 12]> : tensor<4xi64>
    func.return
  }
}

// -----

module @flattened_ids {
  func.func @all_reduce(%operand : tensor<4xi64>) -> tensor<4xi64> {
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
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %inputs1 = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@all_reduce], [@all_reduce]]
    } : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    check.expect_eq_const %results#0, dense<[6, 8, 10, 12]> : tensor<4xi64>
    check.expect_eq_const %results#1, dense<[6, 8, 10, 12]> : tensor<4xi64>
    func.return
  }
}

// -----

module @ragged_replica_groups {
  func.func @all_reduce(%operand : tensor<4xi64>) -> tensor<4xi64> {
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
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %inputs1 = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
    %inputs2 = stablehlo.constant dense<[6, 8, 10, 12]> : tensor<4xi64>
    %results:3 = "interpreter.run_parallel"(%inputs0, %inputs1, %inputs2) {
      programs=[[@all_reduce], [@all_reduce], [@all_reduce]]
    } : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) ->
        (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)
    check.expect_eq_const %results#0, dense<[6, 8, 10, 12]> : tensor<4xi64>
    check.expect_eq_const %results#1, dense<[6, 8, 10, 12]> : tensor<4xi64>
    check.expect_eq_const %results#2, dense<[6, 8, 10, 12]> : tensor<4xi64>
    func.return
  }
}

// -----

module @cross_replica_variadic {
  func.func @all_reduce(%operand0 : tensor<4xi64>, %operand1 : tensor<5xi64>) -> (tensor<4xi64>, tensor<5xi64>) {
    %result:2 = "stablehlo.all_reduce"(%operand0, %operand1) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<4xi64>, tensor<5xi64>) -> (tensor<4xi64>, tensor<5xi64>)
    return %result#0, %result#1 : tensor<4xi64>, tensor<5xi64>
  }
  func.func @main() {
    %process0_oper0 = stablehlo.constant dense<[1,  2,  3,  4]> : tensor<4xi64>
    %process0_oper1 = stablehlo.constant dense<[5,  6,  7,  8,  9]> : tensor<5xi64>
    %process1_oper0 = stablehlo.constant dense<[11, 12, 13, 14]> : tensor<4xi64>
    %process1_oper1 = stablehlo.constant dense<[15, 16, 17, 18, 19]> : tensor<5xi64>
    %results:4 = "interpreter.run_parallel"(%process0_oper0, %process0_oper1, %process1_oper0, %process1_oper1) {
      programs=[[@all_reduce], [@all_reduce]]
    } : (tensor<4xi64>, tensor<5xi64>, tensor<4xi64>, tensor<5xi64>) -> (tensor<4xi64>, tensor<5xi64>, tensor<4xi64>, tensor<5xi64>)
    check.expect_eq_const %results#0, dense<[12, 14, 16, 18]> : tensor<4xi64> // process0_result0
    check.expect_eq_const %results#1, dense<[20, 22, 24, 26, 28]> : tensor<5xi64> // process0_result1
    check.expect_eq_const %results#2, dense<[12, 14, 16, 18]> : tensor<4xi64> // process1_result0
    check.expect_eq_const %results#3, dense<[20, 22, 24, 26, 28]> : tensor<5xi64> // process1_result1
    func.return
  }
}
