// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x2x3xbf16>, tensor<2x3xbf16>)
    %2 = call @expected() : () -> tensor<1x2x3xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<1x2x3xbf16>, tensor<1xi32>, tensor<2x3xbf16>) -> tensor<1x2x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x2x3xbf16>, tensor<1x2x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x2x3xbf16>, tensor<2x3xbf16>) {
    %0 = stablehlo.constant dense<[[[1.773440e+00, 2.890630e-01, -1.968750e+00], [-3.703130e+00, 1.500000e+00, -3.265630e+00]]]> : tensor<1x2x3xbf16>
    %1 = stablehlo.constant dense<[[3.062500e+00, -6.250000e+00, 1.304690e+00], [-2.046880e+00, -5.062500e+00, -1.460940e+00]]> : tensor<2x3xbf16>
    return %0, %1 : tensor<1x2x3xbf16>, tensor<2x3xbf16>
  }
  func.func private @expected() -> tensor<1x2x3xbf16> {
    %0 = stablehlo.constant dense<[[[4.843750e+00, -5.968750e+00, -6.640630e-01], [-5.750000e+00, -3.562500e+00, -4.718750e+00]]]> : tensor<1x2x3xbf16>
    return %0 : tensor<1x2x3xbf16>
  }
}

