// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<i8>, tensor<2x3xi8>, tensor<i8>)
    %1 = call @expected() : () -> tensor<2x3xi8>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<i8>) -> tensor<2x3xi8>
    %3 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<i8>) -> tensor<2x3xi8>
    %4 = stablehlo.clamp %2, %0#1, %3 : tensor<2x3xi8>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<2x3xi8>, tensor<2x3xi8>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<i8>, tensor<2x3xi8>, tensor<i8>) {
    %0 = stablehlo.constant dense<[[2, 0, -1], [2, -1, 1]]> : tensor<2x3xi8>
    %1 = stablehlo.constant dense<-3> : tensor<i8>
    %2 = stablehlo.constant dense<1> : tensor<i8>
    return %1, %0, %2 : tensor<i8>, tensor<2x3xi8>, tensor<i8>
  }
  func.func private @expected() -> tensor<2x3xi8> {
    %0 = stablehlo.constant dense<[[1, 0, -1], [1, -1, 1]]> : tensor<2x3xi8>
    return %0 : tensor<2x3xi8>
  }
}
