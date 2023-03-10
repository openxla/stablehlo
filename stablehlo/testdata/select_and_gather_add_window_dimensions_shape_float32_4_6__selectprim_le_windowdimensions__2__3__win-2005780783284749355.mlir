// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x6xf32>)
    %1 = call @expected() : () -> tensor<3x4xf32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>):
      %6 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<f32>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      stablehlo.return %7, %8 : tensor<f32>, tensor<f32>
    }) {window_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<3x4xf32>, tensor<3x4xf32>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf32>, tensor<4x6xf32>) {
    %0 = stablehlo.constant dense<[[4.12896824, -6.35531473, -1.84030449, 6.9723525, 0.987620532, 1.04766834], [-2.63537812, -3.88554502, 4.46814966, 1.24296653, -7.27006197, 2.14854646], [-2.67894173, 0.0604796484, 0.986472129, 0.451936811, -3.74108315, -1.31748486], [0.206996277, 0.353111774, -3.56800961, -3.42363334, -1.97294855, 2.51790261]]> : tensor<4x6xf32>
    %1 = stablehlo.constant dense<[[2.649370e+00, 5.2231369, 1.26284087, 3.93941283, 0.0840502381, -1.45384884], [-1.33891213, 1.351160e+00, -1.69344187, 2.34142947, -4.06420183, 3.86017942], [-4.25668955, 0.193066373, 2.80178046, 3.54243374, -1.95742321, 0.93727082], [2.67722869, -1.57110488, -3.61369729, -1.95107746, -1.3471787, -1.5559988]]> : tensor<4x6xf32>
    return %0, %1 : tensor<4x6xf32>, tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x4xf32> {
    %0 = stablehlo.constant dense<[[4.46814966, 4.46814966, -7.27006197, -7.27006197], [-2.67894173, 4.46814966, -7.27006197, -7.27006197], [-2.67894173, -3.56800961, -3.56800961, -3.74108315]]> : tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}

