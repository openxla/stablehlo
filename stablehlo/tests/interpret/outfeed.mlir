// RUN: stablehlo-translate --interpret -split-input-file %s

module @distribution_ops {
  func.func @outfeed(%inputs0 : tensor<2x2x2xi64>, %token : !stablehlo.token) -> !stablehlo.token {
    %result = "stablehlo.outfeed"(%inputs0, %token) :
        (tensor<2x2x2xi64>, !stablehlo.token) -> !stablehlo.token
    func.return %result : !stablehlo.token
  }
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[[[1, 2], [3, 4]],
                                         [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
    %token = stablehlo.after_all : !stablehlo.token
    %results0 = "interpreter.run_parallel"(%inputs0, %token) {
      programs=[[@outfeed]]
    } : (tensor<2x2x2xi64>, !stablehlo.token) -> !stablehlo.token
    func.return
  }
}
