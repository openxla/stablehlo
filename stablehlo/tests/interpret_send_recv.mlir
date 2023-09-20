// RUN: stablehlo-translate --interpret -split-input-file %s

module @distribution_ops {
  func.func public @send(%operand : tensor<2x2xi64>, %token : !stablehlo.token) -> !stablehlo.token {
    %result = "stablehlo.send"(%operand, %token) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
      is_host_transfer = true
    } : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
    return %result : !stablehlo.token
  }
  func.func public @recv(%token : !stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token) {
    %results0, %results1 = "stablehlo.recv"(%token) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>,
      is_host_transfer = true
    } : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
    return %results0, %results1 : tensor<2x2xi64>, !stablehlo.token
  }
  func.func public @main() {
    %0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    %1 = stablehlo.after_all : !stablehlo.token
    %2:3 = "interpreter.run_parallel"(%0, %1, %1) {
      programs=[[@send], [@recv]]
    } : (tensor<2x2xi64>, !stablehlo.token, !stablehlo.token) ->
        (!stablehlo.token, tensor<2x2xi64>, !stablehlo.token)
    check.expect_eq_const %2#1, dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    func.return
  }
}
