// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xbf16>, tensor<4x6xbf16>)
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %2 = stablehlo.constant dense<0x7F80> : tensor<bf16>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<bf16>, %arg3: tensor<bf16>):
      %6 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<bf16>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<bf16>
      stablehlo.return %7, %8 : tensor<bf16>, tensor<bf16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xbf16>, tensor<4x6xbf16>, tensor<bf16>, tensor<bf16>) -> (tensor<3x5xbf16>, tensor<3x5xbf16>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xbf16>, tensor<4x6xbf16>) {
    %0 = stablehlo.constant dense<[[2.156250e+00, 3.171880e+00, 2.421880e+00, -4.968750e+00, -4.125000e+00, 4.023440e-01], [5.625000e+00, 5.656250e+00, 1.777340e-01, 1.531250e+00, 2.984380e+00, 2.468750e+00], [-1.679690e+00, -1.953130e+00, -1.812500e+00, 2.843750e+00, 3.640630e+00, 1.328130e-01], [4.875000e+00, 4.656250e+00, 2.343750e+00, 4.187500e+00, 4.648440e-01, 6.796880e-01]]> : tensor<4x6xbf16>
    %1 = stablehlo.constant dense<[[6.437500e+00, 3.859380e+00, 2.625000e+00, -6.132810e-01, -1.304690e+00, 2.828130e+00], [-2.714840e-01, 4.062500e+00, 1.609380e+00, -5.187500e+00, 3.281250e+00, 4.875000e+00], [-1.539060e+00, 2.031250e+00, 4.687500e+00, -5.843750e+00, -1.343750e+00, 1.125000e+00], [1.328130e-01, 2.312500e+00, -2.109380e+00, -1.554690e+00, -1.023440e+00, -2.597660e-01]]> : tensor<4x6xbf16>
    return %0, %1 : tensor<4x6xbf16>, tensor<4x6xbf16>
  }
  func.func private @expected() -> tensor<3x5xbf16> {
    %0 = stablehlo.constant dense<[[5.625000e+00, 1.777340e-01, 1.531250e+00, 1.531250e+00, -4.125000e+00], [-1.679690e+00, 1.777340e-01, 2.843750e+00, 2.843750e+00, 3.640630e+00], [-1.679690e+00, 2.343750e+00, 2.843750e+00, 2.843750e+00, 3.640630e+00]]> : tensor<3x5xbf16>
    return %0 : tensor<3x5xbf16>
  }
}

