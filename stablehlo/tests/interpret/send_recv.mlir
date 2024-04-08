// RUN: stablehlo-translate --interpret -split-input-file %s

module @sequential_send_recv_same_channel {
  func.func @send(%operand : tensor<2x2xi64>, %token : !stablehlo.token) -> (!stablehlo.token, !stablehlo.token) {
    %result0 = "stablehlo.send"(%operand, %token) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
      is_host_transfer = true
    } : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
    %constant = stablehlo.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    %result1 = "stablehlo.send"(%constant, %token) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
      is_host_transfer = true
    } : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
    return %result0, %result1 : !stablehlo.token, !stablehlo.token
  }
  func.func @recv(%token : !stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token) {
    %results0, %results1 = "stablehlo.recv"(%token) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>,
      is_host_transfer = true
    } : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
    %results2, %results3 = "stablehlo.recv"(%token) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>,
      is_host_transfer = true
    } : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
    return %results0, %results1, %results2, %results3 : tensor<2x2xi64>, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token
  }
  func.func @main() {
    %0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %1 = stablehlo.after_all : !stablehlo.token
    %2:6 = "interpreter.run_parallel"(%0, %1, %1) {
      programs=[[@send], [@recv]]
    } : (tensor<2x2xi64>, !stablehlo.token, !stablehlo.token) ->
        (!stablehlo.token, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token)
    check.expect_eq_const %2#2, dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    check.expect_eq_const %2#4, dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    func.return
  }
}

// -----

module @paralllel_send_recv_different_channels {
  func.func @send0(%operand : tensor<2x2xi64>, %token : !stablehlo.token) -> !stablehlo.token {
    %result = "stablehlo.send"(%operand, %token) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
      is_host_transfer = true
    } : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
    return %result : !stablehlo.token
  }
  func.func @send1(%operand : tensor<2x2xi64>, %token : !stablehlo.token) -> !stablehlo.token {
    %result = "stablehlo.send"(%operand, %token) {
      channel_handle = #stablehlo.channel_handle<handle = 2, type = 2>,
      is_host_transfer = true
    } : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
    return %result : !stablehlo.token
  }
  func.func @recv0(%token : !stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token) {
    %results0, %results1 = "stablehlo.recv"(%token) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>,
      is_host_transfer = true
    } : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
    return %results0, %results1 : tensor<2x2xi64>, !stablehlo.token
  }
  func.func @recv1(%token : !stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token) {
    %results0, %results1 = "stablehlo.recv"(%token) {
      channel_handle = #stablehlo.channel_handle<handle = 2, type = 3>,
      is_host_transfer = true
    } : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
    return %results0, %results1 : tensor<2x2xi64>, !stablehlo.token
  }
  func.func @main() {
    %0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %1 = stablehlo.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    %2 = stablehlo.after_all : !stablehlo.token
    %3:6 = "interpreter.run_parallel"(%0, %2, %2, %1, %2, %2) {
      programs=[[@send0], [@recv0], [@send1], [@recv1]]
    } : (tensor<2x2xi64>, !stablehlo.token, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token, !stablehlo.token) ->
        (!stablehlo.token, tensor<2x2xi64>, !stablehlo.token, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token)
    check.expect_eq_const %3#1, dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    check.expect_eq_const %3#4, dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    func.return
  }
}
