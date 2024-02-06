// RUN: stablehlo-translate --interpret -split-input-file %s

module @cross_replica {
  func.func @reduce_scatter(%operand : tensor<2x4xi64>) -> tensor<2x2xi64> {
    %result = "stablehlo.reduce_scatter"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4xi64>) -> tensor<2x2xi64>
    return %result : tensor<2x2xi64>
  }
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[[1, 2, 3, 4],
                                         [5, 6, 7, 8]]> : tensor<2x4xi64>
    %inputs1 = stablehlo.constant dense<[[9, 10, 11, 12],
                                         [13, 14, 15, 16]]> : tensor<2x4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@reduce_scatter], [@reduce_scatter]]
    } : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>)
    check.expect_eq_const %results#0, dense<[[10, 12],
                                             [18, 20]]> : tensor<2x2xi64>
    check.expect_eq_const %results#1, dense<[[14, 16],
                                             [22, 24]]> : tensor<2x2xi64>
    func.return
  }
}

// -----

module @cross_replica_and_partition {
  func.func @reduce_scatter(%operand : tensor<2x4xi64>) -> tensor<2x2xi64> {
    %result = "stablehlo.reduce_scatter"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<2x4xi64>) -> tensor<2x2xi64>
    return %result : tensor<2x2xi64>
  }
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[[1, 2, 3, 4],
                                         [5, 6, 7, 8]]> : tensor<2x4xi64>
    %inputs1 = stablehlo.constant dense<[[9, 10, 11, 12],
                                         [13, 14, 15, 16]]> : tensor<2x4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@reduce_scatter], [@reduce_scatter]]
    } : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>)
    check.expect_eq_const %results#0, dense<[[10, 12],
                                             [18, 20]]> : tensor<2x2xi64>
    check.expect_eq_const %results#1, dense<[[14, 16],
                                             [22, 24]]> : tensor<2x2xi64>
    func.return
  }
}

// -----

module @flattened_ids {
  func.func @reduce_scatter(%operand : tensor<2x4xi64>) -> tensor<2x2xi64> {
    %result = "stablehlo.reduce_scatter"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids
    } : (tensor<2x4xi64>) -> tensor<2x2xi64>
    return %result : tensor<2x2xi64>
  }
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[[1, 2, 3, 4],
                                         [5, 6, 7, 8]]> : tensor<2x4xi64>
    %inputs1 = stablehlo.constant dense<[[9, 10, 11, 12],
                                         [13, 14, 15, 16]]> : tensor<2x4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@reduce_scatter], [@reduce_scatter]]
    } : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>)
    check.expect_eq_const %results#0, dense<[[10, 12],
                                             [18, 20]]> : tensor<2x2xi64>
    check.expect_eq_const %results#1, dense<[[14, 16],
                                             [22, 24]]> : tensor<2x2xi64>
    func.return
  }
}
