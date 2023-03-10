// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<15xf16>
    %1 = call @expected() : () -> tensor<i32>
    %2 = call @argmax(%0) : (tensor<15xf16>) -> tensor<i32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<15xf16> {
    %0 = stablehlo.constant dense<[-1.274410e-01, -2.035160e+00, 3.562500e+00, 1.302730e+00, -1.700200e+00, 2.427980e-01, 4.882810e+00, 1.940430e+00, 1.325200e+00, 3.835450e-01, 3.199220e+00, -1.027340e+00, 4.875490e-01, -1.317380e+00, -1.650390e+00]> : tensor<15xf16>
    return %0 : tensor<15xf16>
  }
  func.func private @expected() -> tensor<i32> {
    %0 = stablehlo.constant dense<6> : tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @argmax(%arg0: tensor<15xf16>) -> tensor<i32> {
    %0 = stablehlo.iota dim = 0 : tensor<15xi32>
    %1 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<15xf16>, tensor<15xi32>, tensor<f16>, tensor<i32>) -> (tensor<f16>, tensor<i32>)
     reducer(%arg1: tensor<f16>, %arg3: tensor<f16>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %4 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %5 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %6 = stablehlo.or %4, %5 : tensor<i1>
      %7 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %8 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = stablehlo.and %7, %8 : tensor<i1>
      %10 = stablehlo.or %6, %9 : tensor<i1>
      %11 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f16>
      %12 = stablehlo.select %10, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %11, %12 : tensor<f16>, tensor<i32>
    }
    return %3#1 : tensor<i32>
  }
}
