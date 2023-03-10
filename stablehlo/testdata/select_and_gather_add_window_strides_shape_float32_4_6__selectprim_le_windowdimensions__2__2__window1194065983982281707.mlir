// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x6xf32>)
    %1 = call @expected() : () -> tensor<2x2xf32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>):
      %6 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<f32>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      stablehlo.return %7, %8 : tensor<f32>, tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf32>, tensor<4x6xf32>) {
    %0 = stablehlo.constant dense<[[-0.253291517, 3.09949613, 4.74506521, -3.27567124, -1.12874556, -5.22338486], [3.75831485, 3.46641779, 2.94156098, -2.29003024, 2.01775813, 1.48182929], [-0.79529494, -2.9649415, -1.78113234, -1.58969486, -1.68874991, 1.392350e-01], [-0.192338094, 1.96468532, 5.83799744, 0.0945905223, 3.20095181, -4.714480e+00]]> : tensor<4x6xf32>
    %1 = stablehlo.constant dense<[[1.14985859, -2.45449471, -3.64832306, -1.45490038, -3.13377285, 0.916883766], [0.0894705504, -0.527379274, 1.99145365, -2.40219951, 0.326169938, 4.40357828], [4.98546457, 1.60041416, 8.282100e-03, 0.560061336, -0.653326511, -2.86092925], [2.35465145, 4.83928871, 2.48427749, 3.86492205, 2.92822051, -0.681164384]]> : tensor<4x6xf32>
    return %0, %1 : tensor<4x6xf32>, tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<2x2xf32> {
    %0 = stablehlo.constant dense<[[3.09949613, -1.12874556], [-2.9649415, -1.68874991]]> : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

