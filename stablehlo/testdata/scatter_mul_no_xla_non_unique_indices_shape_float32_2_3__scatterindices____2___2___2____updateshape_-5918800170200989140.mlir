// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<2> : tensor<1x3x1xi32>
    %1:2 = call @inputs() : () -> (tensor<2x3xf32>, tensor<2x1x3xf32>)
    %2 = call @expected() : () -> tensor<2x3xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>} : (tensor<2x3xf32>, tensor<1x3x1xi32>, tensor<2x1x3xf32>) -> tensor<2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf32>, tensor<2x1x3xf32>) {
    %0 = stablehlo.constant dense<[[-0.307672322, -7.04501772, -2.79272938], [-7.642760e+00, -0.376909941, 1.90229678]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<[[[0.653842985, -2.041810e+00, 1.19794738]], [[3.70987725, 5.58079529, -0.930267572]]]> : tensor<2x1x3xf32>
    return %0, %1 : tensor<2x3xf32>, tensor<2x1x3xf32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[-0.307672322, -7.04501772, 4.46637726], [-7.642760e+00, -0.376909941, -36.6388474]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

