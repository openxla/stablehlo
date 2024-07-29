// RUN: stablehlo-translate --interpret -split-input-file %s

module @cross_replica {
  func.func @all_to_all(%operand : tensor<2x4xi64>) -> tensor<4x2xi64> {
    %result = "stablehlo.all_to_all"(%operand) {
      split_dimension = 1 : i64,
      concat_dimension = 0 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4xi64>) -> tensor<4x2xi64>
    return %result : tensor<4x2xi64>
  }
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[[1, 2, 3, 4],
                                         [5, 6, 7, 8]]> : tensor<2x4xi64>
    %inputs1 = stablehlo.constant dense<[[9, 10, 11, 12],
                                         [13, 14, 15, 16]]> : tensor<2x4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@all_to_all], [@all_to_all]]
    } : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>)
    check.expect_eq_const %results#0, dense<[[1, 2],
                                             [5, 6],
                                             [9, 10],
                                             [13, 14]]> : tensor<4x2xi64>
    check.expect_eq_const %results#1, dense<[[3, 4],
                                             [7, 8],
                                             [11, 12],
                                             [15, 16]]> : tensor<4x2xi64>
    func.return
  }
}

// -----

module @cross_replica_issue_2433 {
  func.func @all_to_all(%operand : tensor<2x4xi64>) -> tensor<4x2xi64> {
    %result = "stablehlo.all_to_all"(%operand) {
      split_dimension = 1 : i64,
      concat_dimension = 0 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[1, 0]]> : tensor<1x2xi64>
    } : (tensor<2x4xi64>) -> tensor<4x2xi64>
    return %result : tensor<4x2xi64>
  }
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[[1, 2, 3, 4],
                                         [5, 6, 7, 8]]> : tensor<2x4xi64>
    %inputs1 = stablehlo.constant dense<[[9, 10, 11, 12],
                                         [13, 14, 15, 16]]> : tensor<2x4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@all_to_all], [@all_to_all]]
    } : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>)
    check.expect_eq_const %results#0, dense<[[11, 12],
                                             [15, 16],
                                             [3, 4],
                                             [7, 8]]> : tensor<4x2xi64>
    check.expect_eq_const %results#1, dense<[[9, 10],
                                             [13, 14],
                                             [1, 2],
                                             [5, 6]]> : tensor<4x2xi64>
    func.return
  }
}

// -----

module @cross_partition {
  func.func @all_to_all(%operand : tensor<2x4xi64>) -> tensor<4x2xi64> {
    %result = "stablehlo.all_to_all"(%operand) {
      split_dimension = 1 : i64,
      concat_dimension = 0 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<2x4xi64>) -> tensor<4x2xi64>
    return %result : tensor<4x2xi64>
  }
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[[1, 2, 3, 4],
                                         [5, 6, 7, 8]]> : tensor<2x4xi64>
    %inputs1 = stablehlo.constant dense<[[9, 10, 11, 12],
                                         [13, 14, 15, 16]]> : tensor<2x4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@all_to_all, @all_to_all]]
    } : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>)
    check.expect_eq_const %results#0, dense<[[1, 2],
                                             [5, 6],
                                             [9, 10],
                                             [13, 14]]> : tensor<4x2xi64>
    check.expect_eq_const %results#1, dense<[[3, 4],
                                             [7, 8],
                                             [11, 12],
                                             [15, 16]]> : tensor<4x2xi64>
    func.return
  }
}

// -----

module @same_split_concat_dim {
  func.func @all_to_all(%operand : tensor<2x4xi64>) -> tensor<2x4xi64> {
    %result = "stablehlo.all_to_all"(%operand) {
      split_dimension = 0 : i64,
      concat_dimension = 0 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4xi64>) -> tensor<2x4xi64>
    return %result : tensor<2x4xi64>
  }
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[[1, 2, 3, 4],
                                         [5, 6, 7, 8]]> : tensor<2x4xi64>
    %inputs1 = stablehlo.constant dense<[[9, 10, 11, 12],
                                         [13, 14, 15, 16]]> : tensor<2x4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@all_to_all], [@all_to_all]]
    } : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
    check.expect_eq_const %results#0, dense<[[1, 2, 3, 4],
                                             [9, 10, 11, 12]]> : tensor<2x4xi64>
    check.expect_eq_const %results#1, dense<[[5, 6, 7, 8],
                                             [13, 14, 15, 16]]> : tensor<2x4xi64>
    func.return
  }
}

// -----

module @cross_replica_variaidic {
  func.func @all_to_all(%operand0 : tensor<2x4xi64>, %operand1 : tensor<3x4xi32>) -> (tensor<4x2xi64>,  tensor<6x2xi32>) {
    %result:2 = "stablehlo.all_to_all"(%operand0, %operand1) {
      split_dimension = 1 : i64,
      concat_dimension = 0 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4xi64>, tensor<3x4xi32>) -> (tensor<4x2xi64>, tensor<6x2xi32>)
    return %result#0, %result#1 : tensor<4x2xi64>, tensor<6x2xi32>
  }
  func.func @main() -> (tensor<4x2xi64>, tensor<6x2xi32>, tensor<4x2xi64>, tensor<6x2xi32>) {
    %process0_oper0 = stablehlo.constant dense<[[1, 2, 3, 4],
                                                [5, 6, 7, 8]]> : tensor<2x4xi64>
    %process0_oper1 = stablehlo.constant dense<[[9, 10, 11, 12],
                                                [13, 14, 15, 16],
                                                [17, 18, 19, 20]]> : tensor<3x4xi32>
    %process1_oper0 = stablehlo.constant dense<[[31, 32, 33, 34],
                                                [35, 36, 37, 38]]> : tensor<2x4xi64>
    %process1_oper1 = stablehlo.constant dense<[[43, 44, 45, 46],
                                                [49, 50, 51, 52],
                                                [53, 54, 55, 56]]> : tensor<3x4xi32>
    %results:4 = "interpreter.run_parallel"(%process0_oper0, %process0_oper1, %process1_oper0, %process1_oper1) {
      programs=[[@all_to_all], [@all_to_all]]
    } : (tensor<2x4xi64>, tensor<3x4xi32>, tensor<2x4xi64>, tensor<3x4xi32>) -> (tensor<4x2xi64>, tensor<6x2xi32>, tensor<4x2xi64>, tensor<6x2xi32>)
    check.expect_eq_const %results#0, dense<[[1, 2],
                                             [5, 6],
                                             [31, 32],
                                             [35, 36]]> : tensor<4x2xi64>
    check.expect_eq_const %results#1, dense<[[9, 10],
                                            [13, 14],
                                            [17, 18],
                                            [43, 44],
                                            [49, 50],
                                            [53, 54]]> : tensor<6x2xi32>
    check.expect_eq_const %results#2, dense<[[3, 4],
                                             [7, 8],
                                             [33, 34],
                                             [37, 38]]> : tensor<4x2xi64>
    check.expect_eq_const %results#3, dense<[[11, 12],
                                             [15, 16],
                                             [19, 20],
                                             [45, 46],
                                             [51, 52],
                                             [55, 56]]> : tensor<6x2xi32>
    func.return %results#0, %results#1, %results#2, %results#3 : tensor<4x2xi64>, tensor<6x2xi32>, tensor<4x2xi64>, tensor<6x2xi32>
  }
}
