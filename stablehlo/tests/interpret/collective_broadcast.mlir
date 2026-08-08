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

// -----

// has_dynamic_root=true: root index 1 selects process 1 as the source, so
// both processes receive process 1's value [[3, 4]].
module @has_dynamic_root {
  func.func @collective_broadcast(%operand: tensor<1x2xi64>,
                                  %root: tensor<1xi32>) -> tensor<1x2xi64> {
    %result = "stablehlo.collective_broadcast"(%operand, %root) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      has_dynamic_root
    } : (tensor<1x2xi64>, tensor<1xi32>) -> tensor<1x2xi64>
    func.return %result : tensor<1x2xi64>
  }
  func.func @main() {
    %operand0 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi64>
    %operand1 = stablehlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
    // root[0] = 1 → source is process_groups[0, 1] = process 1.
    %root = stablehlo.constant dense<[1]> : tensor<1xi32>
    // Inputs ordered per-process: [p0_operand, p0_root, p1_operand, p1_root].
    %results:2 = "interpreter.run_parallel"(%operand0, %root, %operand1, %root) {
      programs = [[@collective_broadcast], [@collective_broadcast]]
    } : (tensor<1x2xi64>, tensor<1xi32>, tensor<1x2xi64>, tensor<1xi32>)
        -> (tensor<1x2xi64>, tensor<1x2xi64>)
    check.expect_eq_const %results#0, dense<[[3, 4]]> : tensor<1x2xi64>
    check.expect_eq_const %results#1, dense<[[3, 4]]> : tensor<1x2xi64>
    func.return
  }
}

// -----

// has_dynamic_root=true variadic: two data operands broadcast from different
// roots. root=[0,1] means data0 comes from process 0 and data1 from process 1.
module @has_dynamic_root_variadic {
  func.func @collective_broadcast(%data0: tensor<1x2xi64>,
                                  %data1: tensor<1x2xi64>,
                                  %root: tensor<2xi32>)
      -> (tensor<1x2xi64>, tensor<1x2xi64>) {
    %result:2 = "stablehlo.collective_broadcast"(%data0, %data1, %root) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      has_dynamic_root
    } : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<2xi32>)
        -> (tensor<1x2xi64>, tensor<1x2xi64>)
    func.return %result#0, %result#1 : tensor<1x2xi64>, tensor<1x2xi64>
  }
  func.func @main() {
    %p0_data0 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi64>
    %p0_data1 = stablehlo.constant dense<[[5, 6]]> : tensor<1x2xi64>
    %p1_data0 = stablehlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
    %p1_data1 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
    // root[0]=0 → data0 from process 0; root[1]=1 → data1 from process 1.
    %root = stablehlo.constant dense<[0, 1]> : tensor<2xi32>
    // Inputs ordered per-process: [p0_d0, p0_d1, p0_root, p1_d0, p1_d1, p1_root].
    %results:4 = "interpreter.run_parallel"(
        %p0_data0, %p0_data1, %root, %p1_data0, %p1_data1, %root) {
      programs = [[@collective_broadcast], [@collective_broadcast]]
    } : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<2xi32>,
         tensor<1x2xi64>, tensor<1x2xi64>, tensor<2xi32>)
        -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    // Both processes receive: data0 from process 0, data1 from process 1.
    check.expect_eq_const %results#0, dense<[[1, 2]]> : tensor<1x2xi64>
    check.expect_eq_const %results#1, dense<[[7, 8]]> : tensor<1x2xi64>
    check.expect_eq_const %results#2, dense<[[1, 2]]> : tensor<1x2xi64>
    check.expect_eq_const %results#3, dense<[[7, 8]]> : tensor<1x2xi64>
    func.return
  }
}
