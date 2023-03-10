// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x6xf32>)
    %1 = call @expected() : () -> tensor<6x15xf32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>):
      %6 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<f32>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      stablehlo.return %7, %8 : tensor<f32>, tensor<f32>
    }) {base_dilations = dense<[2, 3]> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<6x15xf32>, tensor<6x15xf32>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<6x15xf32>, tensor<6x15xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf32>, tensor<4x6xf32>) {
    %0 = stablehlo.constant dense<[[-0.606977522, -4.5480895, 0.542337954, 4.25982714, 3.97463489, 0.172819853], [3.83043742, 8.82498264, -2.31737208, 2.94236755, -4.74602747, 0.483942509], [-0.546568751, -3.48531628, 5.17148495, 0.888249218, 2.20999074, 1.8453964], [0.024966592, -3.73834133, 2.53155947, -4.35680532, -0.585888863, 0.937189817]]> : tensor<4x6xf32>
    %1 = stablehlo.constant dense<[[-4.5223031, -4.02504587, -4.47181749, 1.63240731, -0.189210653, -4.704216], [-5.554750e-01, 3.54502368, 0.73967415, -0.967720627, -0.714872122, -1.24067867], [-1.11142647, 0.0868716165, -0.238088042, -5.24266672, -0.846836984, 3.25735259], [1.10032511, -0.681419492, 3.713760e+00, -1.8503021, -1.63594246, -0.187650815]]> : tensor<4x6xf32>
    return %0, %1 : tensor<4x6xf32>, tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<6x15xf32> {
    %0 = stablehlo.constant dense<[[-0.606977522, 0.000000e+00, -4.5480895, -4.5480895, 0.000000e+00, 0.542337954, 0.542337954, 0.000000e+00, 4.25982714, 4.25982714, 0.000000e+00, 3.97463489, 3.97463489, 0.000000e+00, 0.172819853], [3.83043742, 0.000000e+00, 8.82498264, 8.82498264, 0.000000e+00, -2.31737208, -2.31737208, 0.000000e+00, 2.94236755, 2.94236755, 0.000000e+00, -4.74602747, -4.74602747, 0.000000e+00, 0.483942509], [3.83043742, 0.000000e+00, 8.82498264, 8.82498264, 0.000000e+00, -2.31737208, -2.31737208, 0.000000e+00, 2.94236755, 2.94236755, 0.000000e+00, -4.74602747, -4.74602747, 0.000000e+00, 0.483942509], [-0.546568751, 0.000000e+00, -3.48531628, -3.48531628, 0.000000e+00, 5.17148495, 5.17148495, 0.000000e+00, 0.888249218, 0.888249218, 0.000000e+00, 2.20999074, 2.20999074, 0.000000e+00, 1.8453964], [-0.546568751, 0.000000e+00, -3.48531628, -3.48531628, 0.000000e+00, 5.17148495, 5.17148495, 0.000000e+00, 0.888249218, 0.888249218, 0.000000e+00, 2.20999074, 2.20999074, 0.000000e+00, 1.8453964], [0.024966592, 0.000000e+00, -3.73834133, -3.73834133, 0.000000e+00, 2.53155947, 2.53155947, 0.000000e+00, -4.35680532, -4.35680532, 0.000000e+00, -0.585888863, -0.585888863, 0.000000e+00, 0.937189817]]> : tensor<6x15xf32>
    return %0 : tensor<6x15xf32>
  }
}

