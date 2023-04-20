// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x2x3xf16>, tensor<1x3xf16>)
    %2 = call @expected() : () -> tensor<1x2x3xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x2x3xf16>, tensor<1xi32>, tensor<1x3xf16>) -> tensor<1x2x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x2x3xf16>, tensor<1x2x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x2x3xf16>, tensor<1x3xf16>) {
    %0 = stablehlo.constant dense<[[[-1.101560e+00, -3.882810e+00, 2.164060e+00], [-2.121090e+00, 2.537110e+00, -1.364260e+00]]]> : tensor<1x2x3xf16>
    %1 = stablehlo.constant dense<[[9.484860e-02, -2.328130e+00, 1.257810e+00]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x2x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x2x3xf16> {
    %0 = stablehlo.constant dense<[[[-1.101560e+00, -3.882810e+00, 2.164060e+00], [-2.011720e-01, -5.906250e+00, -1.715820e+00]]]> : tensor<1x2x3xf16>
    return %0 : tensor<1x2x3xf16>
  }
}

