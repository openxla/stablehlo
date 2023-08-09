// RUN: stablehlo-translate --interpret -split-input-file %s

module @distribution_ops {
  func.func public @outfeed(%inputs0 : tensor<2x2x2xi64>, %token : !stablehlo.token) -> !stablehlo.token {
    %result = "stablehlo.outfeed"(%inputs0, %token) {
      outfeed_config = ""
    } : (tensor<2x2x2xi64>, !stablehlo.token) -> !stablehlo.token
    func.return %result : !stablehlo.token
  }
  func.func public @main() {
    %inputs0 = stablehlo.constant dense<[[[1, 2], [3, 4]],
                                         [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
    %token = stablehlo.after_all : !stablehlo.token
    %result = "interpreter.run_parallel"(%inputs0, %token) {
      programs=[["outfeed"]]
    } : (tensor<2x2x2xi64>, !stablehlo.token) -> !stablehlo.token
    func.return
  }
}
