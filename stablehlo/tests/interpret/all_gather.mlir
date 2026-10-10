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

// -----

module @mesh_axes_subaxis {
  func.func @all_gather(%operand : tensor<1xi64>) -> tensor<2xi64> {
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 0 : i64,
      replica_groups = #stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = "x", size = 2>, #stablehlo.mesh_axis<name = "y", size = 4>]>, axes = [#stablehlo.axis_ref<name = "y", sub_axis_info = (1)2>]>
    } : (tensor<1xi64>) -> tensor<2xi64>
    return %result : tensor<2xi64>
  }
  func.func @main() {
    %p0 = stablehlo.constant dense<[0]> : tensor<1xi64>
    %p1 = stablehlo.constant dense<[1]> : tensor<1xi64>
    %p2 = stablehlo.constant dense<[2]> : tensor<1xi64>
    %p3 = stablehlo.constant dense<[3]> : tensor<1xi64>
    %p4 = stablehlo.constant dense<[4]> : tensor<1xi64>
    %p5 = stablehlo.constant dense<[5]> : tensor<1xi64>
    %p6 = stablehlo.constant dense<[6]> : tensor<1xi64>
    %p7 = stablehlo.constant dense<[7]> : tensor<1xi64>
    %results:8 = "interpreter.run_parallel"(%p0, %p1, %p2, %p3, %p4, %p5, %p6, %p7) {
      programs=[[@all_gather], [@all_gather], [@all_gather], [@all_gather],
                [@all_gather], [@all_gather], [@all_gather], [@all_gather]]
    } : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
         tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) ->
        (tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>,
         tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>)
    check.expect_eq_const %results#0, dense<[0, 2]> : tensor<2xi64>
    check.expect_eq_const %results#1, dense<[1, 3]> : tensor<2xi64>
    check.expect_eq_const %results#2, dense<[0, 2]> : tensor<2xi64>
    check.expect_eq_const %results#3, dense<[1, 3]> : tensor<2xi64>
    check.expect_eq_const %results#4, dense<[4, 6]> : tensor<2xi64>
    check.expect_eq_const %results#5, dense<[5, 7]> : tensor<2xi64>
    check.expect_eq_const %results#6, dense<[4, 6]> : tensor<2xi64>
    check.expect_eq_const %results#7, dense<[5, 7]> : tensor<2xi64>
    func.return
  }
}

// -----

// Four all_gather ops in a row share one process group and one channel, so they
// all rendezvous on the same channel. Each process starts from a distinct power
// of ten and every round adds process 7's running value, so no two rounds
// contribute the same values and a process that picks up a contribution left
// over from an earlier round reaches a different result.
module @repeated_rendezvous_on_one_channel {
  func.func @all_gather(%operand : tensor<1xi64>) -> tensor<1xi64> {
    %gather0 = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>
    } : (tensor<1xi64>) -> tensor<8xi64>
    %peer0 = stablehlo.slice %gather0 [7:8] : (tensor<8xi64>) -> tensor<1xi64>
    %state0 = stablehlo.add %peer0, %operand : tensor<1xi64>

    %gather1 = "stablehlo.all_gather"(%state0) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>
    } : (tensor<1xi64>) -> tensor<8xi64>
    %peer1 = stablehlo.slice %gather1 [7:8] : (tensor<8xi64>) -> tensor<1xi64>
    %state1 = stablehlo.add %peer1, %operand : tensor<1xi64>

    %gather2 = "stablehlo.all_gather"(%state1) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>
    } : (tensor<1xi64>) -> tensor<8xi64>
    %peer2 = stablehlo.slice %gather2 [7:8] : (tensor<8xi64>) -> tensor<1xi64>
    %state2 = stablehlo.add %peer2, %operand : tensor<1xi64>

    %gather3 = "stablehlo.all_gather"(%state2) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>
    } : (tensor<1xi64>) -> tensor<8xi64>
    %peer3 = stablehlo.slice %gather3 [7:8] : (tensor<8xi64>) -> tensor<1xi64>
    %state3 = stablehlo.add %peer3, %operand : tensor<1xi64>

    return %state3 : tensor<1xi64>
  }
  func.func @main() {
    %p0 = stablehlo.constant dense<[1]> : tensor<1xi64>
    %p1 = stablehlo.constant dense<[10]> : tensor<1xi64>
    %p2 = stablehlo.constant dense<[100]> : tensor<1xi64>
    %p3 = stablehlo.constant dense<[1000]> : tensor<1xi64>
    %p4 = stablehlo.constant dense<[10000]> : tensor<1xi64>
    %p5 = stablehlo.constant dense<[100000]> : tensor<1xi64>
    %p6 = stablehlo.constant dense<[1000000]> : tensor<1xi64>
    %p7 = stablehlo.constant dense<[10000000]> : tensor<1xi64>
    %results:8 = "interpreter.run_parallel"(%p0, %p1, %p2, %p3, %p4, %p5, %p6, %p7) {
      programs=[[@all_gather], [@all_gather], [@all_gather], [@all_gather],
                [@all_gather], [@all_gather], [@all_gather], [@all_gather]]
    } : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
         tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) ->
        (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
         tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
    check.expect_eq_const %results#0, dense<[40000001]> : tensor<1xi64>
    check.expect_eq_const %results#1, dense<[40000010]> : tensor<1xi64>
    check.expect_eq_const %results#2, dense<[40000100]> : tensor<1xi64>
    check.expect_eq_const %results#3, dense<[40001000]> : tensor<1xi64>
    check.expect_eq_const %results#4, dense<[40010000]> : tensor<1xi64>
    check.expect_eq_const %results#5, dense<[40100000]> : tensor<1xi64>
    check.expect_eq_const %results#6, dense<[41000000]> : tensor<1xi64>
    check.expect_eq_const %results#7, dense<[50000000]> : tensor<1xi64>
    func.return
  }
}
