// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4xf32>, tensor<4x4xf32>)
    %1 = call @expected() : () -> tensor<4xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<4xf32>, tensor<4x4xf32>) -> tensor<4xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4xf32>, tensor<4x4xf32>) {
    %0 = stablehlo.constant dense<[-2.67958117, 4.91505384, -2.93944049, -0.189866632]> : tensor<4xf32>
    %1 = stablehlo.constant dense<[[-0.876051664, 2.81679201, 1.48077691, 1.10807765], [-1.83372617, 1.35355616, 3.68328929, -4.30171204], [-6.15009593, -5.9722824, -0.454436153, -1.66895545], [1.09934378, 5.87006092, -3.10807371, 0.333222806]]> : tensor<4x4xf32>
    return %0, %1 : tensor<4xf32>, tensor<4x4xf32>
  }
  func.func private @expected() -> tensor<4xf32> {
    %0 = stablehlo.constant dense<[11.203701, 15.5456181, 16.0616112, -19.2698021]> : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

