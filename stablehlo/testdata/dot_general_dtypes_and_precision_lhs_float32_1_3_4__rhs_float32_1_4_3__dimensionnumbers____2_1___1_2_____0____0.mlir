// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xf32>, tensor<1x4x3xf32>)
    %1 = call @expected() : () -> tensor<1xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>} : (tensor<1x3x4xf32>, tensor<1x4x3xf32>) -> tensor<1xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xf32>, tensor<1x4x3xf32>) {
    %0 = stablehlo.constant dense<[[[-1.02716792, -0.84180355, -1.21495497, 1.22473526], [1.533764, 3.72483683, -3.63489795, 1.59403718], [-2.40816736, -1.31785822, -1.78571689, 0.202816486]]]> : tensor<1x3x4xf32>
    %1 = stablehlo.constant dense<[[[2.14761162, 2.50920868, -4.64897203], [1.72184336, -1.84444284, -1.14315307], [-1.28549385, 2.20468378, 1.76219583], [-3.49526167, -2.27656484, 2.42410755]]]> : tensor<1x4x3xf32>
    return %0, %1 : tensor<1x3x4xf32>, tensor<1x4x3xf32>
  }
  func.func private @expected() -> tensor<1xf32> {
    %0 = stablehlo.constant dense<-10.9919271> : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

