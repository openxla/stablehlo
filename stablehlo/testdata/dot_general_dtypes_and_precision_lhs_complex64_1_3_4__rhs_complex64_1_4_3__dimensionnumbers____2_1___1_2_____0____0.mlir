// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<1xcomplex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>} : (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>) -> tensor<1xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xcomplex<f32>>, tensor<1xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[[(-1.25878465,0.483840942), (4.48353529,-4.25415182), (-4.04022217,4.19074869), (0.28382197,6.39763308)], [(2.47538543,-1.2548666), (0.612360358,1.41316974), (1.79300344,4.08260965), (2.20243955,-4.90612078)], [(-0.455038577,-0.547110319), (-1.97714114,2.25501418), (6.69369841,4.60687923), (7.83684444,-3.48328114)]]]> : tensor<1x3x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[[(2.49420547,-3.58653378), (3.93156433,-6.49863815), (-2.23460746,-2.46788073)], [(4.53474379,-2.00012064), (-1.716169,-3.13752127), (2.88201332,-2.77150035)], [(-1.66835463,0.221464783), (-4.64653635,3.18384242), (1.73326027,-3.6916244)], [(-6.3931179,0.544783413), (-0.0288959388,1.63391304), (4.93296051,0.870400726)]]]> : tensor<1x4x3xcomplex<f32>>
    return %0, %1 : tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<1xcomplex<f32>> {
    %0 = stablehlo.constant dense<(73.0318756,-118.821198)> : tensor<1xcomplex<f32>>
    return %0 : tensor<1xcomplex<f32>>
  }
}

