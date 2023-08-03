// RUN: stablehlo-translate --interpret -split-input-file %s

module @distribution_ops {
  func.func public @partition_id() -> tensor<ui32> {
    %result = stablehlo.partition_id : tensor<ui32>
    return %result : tensor<ui32>
  }
  func.func public @main() {
    %results:2 = "interpreter.run_parallel"() {
      programs=[["partition_id", "partition_id"]]
    } : () -> (tensor<ui32>, tensor<ui32>)
    check.expect_eq_const %results#0, dense<0> : tensor<ui32>
    check.expect_eq_const %results#1, dense<1> : tensor<ui32>
    func.return
  }
}
