// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[3, 2]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3xf32>, tensor<2xf32>)
    %2 = call @expected() : () -> tensor<4x2x3xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf32>, tensor<2xi32>, tensor<2xf32>) -> tensor<4x2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[[[0.630705654, 1.25725687, 4.59843111], [0.263744563, 4.30864906, -1.15814757]], [[-2.85588813, 4.05002165, 2.65993714], [1.95215988, 3.07178116, -2.0432117]], [[-4.15850449, 1.80068922, 2.70222163], [-0.0853943154, -5.23988295, -1.0240227]], [[3.67574072, -3.22453189, -1.52127862], [-2.89576793, 3.39081883, -0.135205105]]]> : tensor<4x2x3xf32>
    %1 = stablehlo.constant dense<[1.794577, -4.88674307]> : tensor<2xf32>
    return %0, %1 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<4x2x3xf32> {
    %0 = stablehlo.constant dense<[[[0.630705654, 1.25725687, 4.59843111], [0.263744563, 4.30864906, -1.15814757]], [[-2.85588813, 4.05002165, 2.65993714], [1.95215988, 3.07178116, -2.0432117]], [[-4.15850449, 1.80068922, 2.70222163], [-0.0853943154, -5.23988295, -1.0240227]], [[3.67574072, -3.22453189, -1.52127862], [-2.89576793, 3.39081883, -4.88674307]]]> : tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}

