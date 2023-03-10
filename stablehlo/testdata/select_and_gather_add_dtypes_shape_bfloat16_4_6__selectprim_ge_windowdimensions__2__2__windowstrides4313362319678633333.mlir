// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xbf16>, tensor<4x6xbf16>)
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %2 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<bf16>, %arg3: tensor<bf16>):
      %6 = stablehlo.compare  GE, %arg0, %arg2,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<bf16>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<bf16>
      stablehlo.return %7, %8 : tensor<bf16>, tensor<bf16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xbf16>, tensor<4x6xbf16>, tensor<bf16>, tensor<bf16>) -> (tensor<3x5xbf16>, tensor<3x5xbf16>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xbf16>, tensor<4x6xbf16>) {
    %0 = stablehlo.constant dense<[[-1.101560e+00, -8.164060e-01, -1.523440e+00, 1.054690e+00, -4.562500e+00, -3.593750e-01], [4.000000e+00, 9.179680e-01, -5.859380e-01, -8.046880e-01, -1.875000e+00, -5.781250e+00], [-1.687500e+00, -6.906250e+00, 1.046880e+00, -2.250000e+00, -4.812500e+00, 5.125000e+00], [-1.679690e+00, 2.218750e+00, 1.523440e+00, -3.515630e-01, 1.695310e+00, 1.037500e+01]]> : tensor<4x6xbf16>
    %1 = stablehlo.constant dense<[[2.078130e+00, -4.335940e-01, 7.312500e+00, -2.328130e+00, 4.667970e-01, 3.656250e+00], [1.242190e+00, -2.718750e+00, 8.007810e-02, -6.187500e+00, 3.890630e+00, -5.093750e+00], [-5.687500e+00, -5.406250e+00, 4.781250e+00, 1.992190e+00, 3.984380e-01, -8.007810e-02], [2.656250e+00, -5.976560e-01, -3.156250e+00, 8.125000e-01, 1.929690e+00, 2.406250e+00]]> : tensor<4x6xbf16>
    return %0, %1 : tensor<4x6xbf16>, tensor<4x6xbf16>
  }
  func.func private @expected() -> tensor<3x5xbf16> {
    %0 = stablehlo.constant dense<[[-1.101560e+00, -1.523440e+00, -1.523440e+00, -1.875000e+00, -1.875000e+00], [4.000000e+00, 1.046880e+00, 1.046880e+00, -1.875000e+00, -1.875000e+00], [-1.679690e+00, 1.046880e+00, 1.046880e+00, -2.250000e+00, 1.037500e+01]]> : tensor<3x5xbf16>
    return %0 : tensor<3x5xbf16>
  }
}

