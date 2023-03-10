// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3xi16>
    %1 = call @expected() : () -> tensor<1xi16>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xi16>) -> tensor<1xi16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xi16>, tensor<1xi16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3xi16> {
    %0 = stablehlo.constant dense<[4, -1, -2]> : tensor<3xi16>
    return %0 : tensor<3xi16>
  }
  func.func private @expected() -> tensor<1xi16> {
    %0 = stablehlo.constant dense<-1> : tensor<1xi16>
    return %0 : tensor<1xi16>
  }
}
