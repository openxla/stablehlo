// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<15xui16>
    %1 = call @expected() : () -> tensor<i32>
    %2 = call @argmin(%0) : (tensor<15xui16>) -> tensor<i32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<15xui16> {
    %0 = stablehlo.constant dense<[2, 2, 0, 0, 1, 1, 0, 0, 1, 2, 1, 3, 2, 2, 0]> : tensor<15xui16>
    return %0 : tensor<15xui16>
  }
  func.func private @expected() -> tensor<i32> {
    %0 = stablehlo.constant dense<2> : tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @argmin(%arg0: tensor<15xui16>) -> tensor<i32> {
    %0 = stablehlo.iota dim = 0 : tensor<15xi32>
    %1 = stablehlo.constant dense<65535> : tensor<ui16>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<15xui16>, tensor<15xi32>, tensor<ui16>, tensor<i32>) -> (tensor<ui16>, tensor<i32>)
     reducer(%arg1: tensor<ui16>, %arg3: tensor<ui16>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %4 = stablehlo.compare  LT, %arg1, %arg3,  UNSIGNED : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
      %5 = stablehlo.compare  NE, %arg1, %arg1,  UNSIGNED : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
      %6 = stablehlo.or %4, %5 : tensor<i1>
      %7 = stablehlo.compare  EQ, %arg1, %arg3,  UNSIGNED : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
      %8 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = stablehlo.and %7, %8 : tensor<i1>
      %10 = stablehlo.or %6, %9 : tensor<i1>
      %11 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<ui16>
      %12 = stablehlo.select %10, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %11, %12 : tensor<ui16>, tensor<i32>
    }
    return %3#1 : tensor<i32>
  }
}
