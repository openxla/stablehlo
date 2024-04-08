// RUN: stablehlo-translate --interpret -split-input-file %s

module @cross_replica {
  func.func @collective_broadcast(%operand : tensor<1x2xi64>) -> tensor<1x2xi64> {
    %result = "stablehlo.collective_broadcast"(%operand) {
      replica_groups = dense<[[2, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<1x2xi64>) -> tensor<1x2xi64>
    return %result : tensor<1x2xi64>
  }
  func.func @main() {
    %operand0 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi64>
    %operand1 = stablehlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
    %operand2 = stablehlo.constant dense<[[5, 6]]> : tensor<1x2xi64>
    %operand3 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
    %results:4 = "interpreter.run_parallel"(%operand0, %operand1, %operand2, %operand3) {
      programs=[[@collective_broadcast], [@collective_broadcast],
                [@collective_broadcast], [@collective_broadcast]]
    } : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) ->
        (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    check.expect_eq_const %results#0, dense<[[0, 0]]> : tensor<1x2xi64>
    check.expect_eq_const %results#1, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#2, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#3, dense<[[0, 0]]> : tensor<1x2xi64>
    func.return
  }
}

// -----

module @cross_replica_multiple_output {
  func.func @collective_broadcast(%operand : tensor<1x2xi64>) -> tensor<1x2xi64> {
    %result = "stablehlo.collective_broadcast"(%operand) {
      replica_groups = dense<[[2, 1, 0]]> : tensor<1x3xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<1x2xi64>) -> tensor<1x2xi64>
    return %result : tensor<1x2xi64>
  }
  func.func @main() {
    %operand0 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi64>
    %operand1 = stablehlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
    %operand2 = stablehlo.constant dense<[[5, 6]]> : tensor<1x2xi64>
    %operand3 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
    %results:4 = "interpreter.run_parallel"(%operand0, %operand1, %operand2, %operand3) {
      programs=[[@collective_broadcast], [@collective_broadcast],
                [@collective_broadcast], [@collective_broadcast]]
    } : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) ->
        (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    check.expect_eq_const %results#0, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#1, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#2, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#3, dense<[[0, 0]]> : tensor<1x2xi64>
    func.return
  }
}

// -----

module @cross_replica_single_replica {
  func.func @collective_broadcast(%operand : tensor<1x2xi64>) -> tensor<1x2xi64> {
    %result = "stablehlo.collective_broadcast"(%operand) {
      replica_groups = dense<[[0]]> : tensor<1x1xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<1x2xi64>) -> tensor<1x2xi64>
    return %result : tensor<1x2xi64>
  }
  func.func @main() {
    %operand0 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi64>
    %operand1 = stablehlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
    %operand2 = stablehlo.constant dense<[[5, 6]]> : tensor<1x2xi64>
    %operand3 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
    %results:4 = "interpreter.run_parallel"(%operand0, %operand1, %operand2, %operand3) {
      programs=[[@collective_broadcast, @collective_broadcast,
                 @collective_broadcast, @collective_broadcast]]
    } : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) ->
        (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    check.expect_eq_const %results#0, dense<[[1, 2]]> : tensor<1x2xi64>
    check.expect_eq_const %results#1, dense<[[3, 4]]> : tensor<1x2xi64>
    check.expect_eq_const %results#2, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#3, dense<[[7, 8]]> : tensor<1x2xi64>
    func.return
  }
}

// -----

module @cross_replica_multiple_partitions {
  func.func @collective_broadcast(%operand : tensor<1x2xi64>) -> tensor<1x2xi64> {
    %result = "stablehlo.collective_broadcast"(%operand) {
      replica_groups = dense<[[1, 0]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<1x2xi64>) -> tensor<1x2xi64>
    return %result : tensor<1x2xi64>
  }
  func.func @main() {
    %operand0 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi64>
    %operand1 = stablehlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
    %operand2 = stablehlo.constant dense<[[5, 6]]> : tensor<1x2xi64>
    %operand3 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
    %results:4 = "interpreter.run_parallel"(%operand0, %operand1, %operand2, %operand3) {
      programs=[[@collective_broadcast, @collective_broadcast],
                [@collective_broadcast, @collective_broadcast]]
    } : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) ->
        (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    check.expect_eq_const %results#0, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#1, dense<[[7, 8]]> : tensor<1x2xi64>
    check.expect_eq_const %results#2, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#3, dense<[[7, 8]]> : tensor<1x2xi64>
    func.return
  }
}

// -----

module @cross_partition {
  func.func @collective_broadcast(%operand : tensor<1x2xi64>) -> tensor<1x2xi64> {
    %result = "stablehlo.collective_broadcast"(%operand) {
      replica_groups = dense<[[2, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<1x2xi64>) -> tensor<1x2xi64>
    return %result : tensor<1x2xi64>
  }
  func.func @main() {
    %operand0 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi64>
    %operand1 = stablehlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
    %operand2 = stablehlo.constant dense<[[5, 6]]> : tensor<1x2xi64>
    %operand3 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
    %results:4 = "interpreter.run_parallel"(%operand0, %operand1, %operand2, %operand3) {
      programs=[[@collective_broadcast, @collective_broadcast,
                 @collective_broadcast, @collective_broadcast]]
    } : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) ->
        (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    check.expect_eq_const %results#0, dense<[[0, 0]]> : tensor<1x2xi64>
    check.expect_eq_const %results#1, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#2, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#3, dense<[[0, 0]]> : tensor<1x2xi64>
    func.return
  }
}

// -----

module @cross_partition_multiple_output {
  func.func @collective_broadcast(%operand : tensor<1x2xi64>) -> tensor<1x2xi64> {
    %result = "stablehlo.collective_broadcast"(%operand) {
      replica_groups = dense<[[2, 1, 0]]> : tensor<1x3xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<1x2xi64>) -> tensor<1x2xi64>
    return %result : tensor<1x2xi64>
  }
  func.func @main() {
    %operand0 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi64>
    %operand1 = stablehlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
    %operand2 = stablehlo.constant dense<[[5, 6]]> : tensor<1x2xi64>
    %operand3 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
    %results:4 = "interpreter.run_parallel"(%operand0, %operand1, %operand2, %operand3) {
      programs=[[@collective_broadcast, @collective_broadcast,
                 @collective_broadcast, @collective_broadcast]]
    } : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) ->
        (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    check.expect_eq_const %results#0, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#1, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#2, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#3, dense<[[0, 0]]> : tensor<1x2xi64>
    func.return
  }
}

// -----

module @cross_partition_single_partition {
  func.func @collective_broadcast(%operand : tensor<1x2xi64>) -> tensor<1x2xi64> {
    %result = "stablehlo.collective_broadcast"(%operand) {
      replica_groups = dense<[[0]]> : tensor<1x1xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<1x2xi64>) -> tensor<1x2xi64>
    return %result : tensor<1x2xi64>
  }
  func.func @main() {
    %operand0 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi64>
    %operand1 = stablehlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
    %operand2 = stablehlo.constant dense<[[5, 6]]> : tensor<1x2xi64>
    %operand3 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
    %results:4 = "interpreter.run_parallel"(%operand0, %operand1, %operand2, %operand3) {
      programs=[[@collective_broadcast], [@collective_broadcast],
                [@collective_broadcast], [@collective_broadcast]]
    } : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) ->
        (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    check.expect_eq_const %results#0, dense<[[1, 2]]> : tensor<1x2xi64>
    check.expect_eq_const %results#1, dense<[[3, 4]]> : tensor<1x2xi64>
    check.expect_eq_const %results#2, dense<[[5, 6]]> : tensor<1x2xi64>
    check.expect_eq_const %results#3, dense<[[7, 8]]> : tensor<1x2xi64>
    func.return
  }
}

// -----

module @cross_partition_multiple_replicas {
  func.func @collective_broadcast(%operand : tensor<1x2xi64>) -> tensor<1x2xi64> {
    %result = "stablehlo.collective_broadcast"(%operand) {
      replica_groups = dense<[[1, 0]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<1x2xi64>) -> tensor<1x2xi64>
    return %result : tensor<1x2xi64>
  }
  func.func @main() {
    %operand0 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi64>
    %operand1 = stablehlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
    %operand2 = stablehlo.constant dense<[[5, 6]]> : tensor<1x2xi64>
    %operand3 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
    %results:4 = "interpreter.run_parallel"(%operand0, %operand1, %operand2, %operand3) {
      programs=[[@collective_broadcast, @collective_broadcast],
                [@collective_broadcast, @collective_broadcast]]
    } : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) ->
        (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    check.expect_eq_const %results#0, dense<[[3, 4]]> : tensor<1x2xi64>
    check.expect_eq_const %results#1, dense<[[3, 4]]> : tensor<1x2xi64>
    check.expect_eq_const %results#2, dense<[[7, 8]]> : tensor<1x2xi64>
    check.expect_eq_const %results#3, dense<[[7, 8]]> : tensor<1x2xi64>
    func.return
  }
}
