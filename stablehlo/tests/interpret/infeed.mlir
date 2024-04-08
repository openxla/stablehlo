// RUN: stablehlo-translate --interpret -split-input-file %s

module @distribution_ops {
  func.func @infeed(%token : !stablehlo.token) ->
                          (tensor<2x2xi64>, !stablehlo.token,
                           tensor<2x2xi64>, !stablehlo.token) {
    %results0:2 = "stablehlo.infeed"(%token) :
        (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
    %results1:2 = "stablehlo.infeed"(%token) :
        (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
    func.return %results0#0, %results0#1, %results1#0, %results1#1 :
        tensor<2x2xi64>, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token
  }
  func.func @infeed_queue0() -> (tensor<2x2xi64>) {
    %queue0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    func.return %queue0 : tensor<2x2xi64>
  }
  func.func @infeed_queue1() -> (tensor<2x2xi64>) {
    %queue0 = stablehlo.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    func.return %queue0 : tensor<2x2xi64>
  }
  func.func @main() {
    %token = stablehlo.after_all : !stablehlo.token
    %results:4 = "interpreter.run_parallel"(%token) {
      infeed=[@infeed_queue0, @infeed_queue1],
      programs=[[@infeed]]
    } : (!stablehlo.token) ->
        (tensor<2x2xi64>, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token)
    check.expect_eq_const %results#0, dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
    check.expect_eq_const %results#2, dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>
    func.return
  }
}
