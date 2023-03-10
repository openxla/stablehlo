// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x7xf16>
    %1 = call @expected() : () -> tensor<5x7xf16>
    %2 = stablehlo.reduce_precision %0, format = e8m7 : tensor<5x7xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<5x7xf16>, tensor<5x7xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x7xf16> {
    %0 = stablehlo.constant dense<[[-4.980470e+00, -1.423830e+00, -7.729490e-01, 3.022460e-01, -1.522460e+00, -1.368160e+00, 8.823240e-01], [-1.669920e+00, 3.917970e+00, 2.562500e+00, -1.766600e+00, -1.927730e+00, 2.458980e+00, 2.394530e+00], [2.748050e+00, 2.085940e+00, 2.140630e+00, -2.833980e+00, -1.005860e+00, 5.558590e+00, 1.140630e+00], [-4.691410e+00, 5.468750e-01, -5.561520e-01, -1.989260e+00, -1.336910e+00, 5.836480e-04, 5.302730e-01], [-5.058590e+00, -8.310550e-01, 7.670890e-01, -3.335940e+00, -5.917960e+00, -2.404300e+00, -2.906250e+00]]> : tensor<5x7xf16>
    return %0 : tensor<5x7xf16>
  }
  func.func private @expected() -> tensor<5x7xf16> {
    %0 = stablehlo.constant dense<[[-4.968750e+00, -1.421880e+00, -7.734380e-01, 3.027340e-01, -1.523440e+00, -1.367190e+00, 8.828120e-01], [-1.671880e+00, 3.921880e+00, 2.562500e+00, -1.765630e+00, -1.929690e+00, 2.453130e+00, 2.390630e+00], [2.750000e+00, 2.093750e+00, 2.140630e+00, -2.828130e+00, -1.007810e+00, 5.562500e+00, 1.140630e+00], [-4.687500e+00, 5.468750e-01, -5.546880e-01, -1.992190e+00, -1.335940e+00, 5.836480e-04, 5.312500e-01], [-5.062500e+00, -8.320310e-01, 7.656250e-01, -3.343750e+00, -5.906250e+00, -2.406250e+00, -2.906250e+00]]> : tensor<5x7xf16>
    return %0 : tensor<5x7xf16>
  }
}
