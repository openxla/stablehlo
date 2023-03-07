// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xi8>
    %1 = call @expected() : () -> tensor<3xi8>
    %2 = stablehlo.constant dense<0> : tensor<i8>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xi8>, tensor<i8>) -> tensor<3xi8>
     reducer(%arg0: tensor<i8>, %arg1: tensor<i8>)  {
      %5 = stablehlo.add %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xi8>, tensor<3xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xi8> {
    %0 = stablehlo.constant dense<[[-1, 4, 3], [0, 4, 3]]> : tensor<2x3xi8>
    return %0 : tensor<2x3xi8>
  }
  func.func private @expected() -> tensor<3xi8> {
    %0 = stablehlo.constant dense<[-1, 8, 6]> : tensor<3xi8>
    return %0 : tensor<3xi8>
  }
}
