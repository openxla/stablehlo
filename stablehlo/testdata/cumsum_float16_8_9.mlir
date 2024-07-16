// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x9xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<8x9xf16>
    %1 = call @expected() : () -> tensor<8x9xf16>
    %2 = call @cumsum(%0) : (tensor<8x9xf16>) -> tensor<8x9xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<8x9xf16>, tensor<8x9xf16>) -> ()
    return %2 : tensor<8x9xf16>
  }
  func.func private @inputs() -> (tensor<8x9xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.979980e-01, 2.201170e+00, -4.892580e-01, -2.476560e+00, 8.212890e-01, -1.108400e+00, -2.546390e-01, -5.141600e-01, 5.628900e+00], [-4.421880e+00, -2.644530e+00, -1.182620e+00, 1.538090e+00, -3.037110e+00, 4.453130e+00, -1.069640e-02, -2.039060e+00, -2.386470e-01], [-3.283690e-01, -1.354490e+00, 2.863280e+00, 1.047970e-01, -5.820310e-01, 1.229490e+00, 5.601560e+00, -2.806640e+00, -1.148440e+00], [2.107420e+00, -1.015630e+00, -7.359380e+00, -7.042960e+00, 1.913090e+00, -6.292960e+00, -2.773440e+00, 8.706050e-01, 4.174800e-01], [-2.888670e+00, -1.996090e+00, 4.539060e+00, 5.639650e-01, -7.226560e-01, -1.039060e+00, -5.011720e+00, -2.341800e+00, 4.101560e+00], [-4.171880e+00, -1.065430e+00, 1.106450e+00, 2.804690e+00, -4.035160e+00, -4.773440e+00, -2.732420e+00, -2.275390e+00, -2.191410e+00], [2.111330e+00, -4.152830e-01, -3.264160e-01, -1.489260e+00, 1.603520e+00, -1.934570e+00, 2.683590e+00, -2.763670e-01, -2.033200e+00], [-6.386710e+00, -1.771240e-01, 7.011710e-01, -3.185550e+00, 6.025390e-01, -2.093750e+00, 6.141660e-03, 2.443360e+00, 2.972660e+00]]> : tensor<8x9xf16>
    return %cst : tensor<8x9xf16>
  }
  func.func private @expected() -> (tensor<8x9xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.979980e-01, 2.201170e+00, -4.892580e-01, -2.476560e+00, 8.212890e-01, -1.108400e+00, -2.546390e-01, -5.141600e-01, 5.628900e+00], [-4.621090e+00, -4.433590e-01, -1.671880e+00, -9.384760e-01, -2.214840e+00, 3.343750e+00, -2.653810e-01, -2.552730e+00, 5.390630e+00], [-4.949220e+00, -1.797850e+00, 1.191410e+00, -8.334960e-01, -2.796880e+00, 4.574220e+00, 5.335940e+00, -5.359380e+00, 4.242190e+00], [-2.841800e+00, -2.812500e+00, -6.167960e+00, -7.875000e+00, -8.837890e-01, -1.718750e+00, 2.562500e+00, -4.488280e+00, 4.660160e+00], [-5.730460e+00, -4.808590e+00, -1.628910e+00, -7.312500e+00, -1.606450e+00, -2.757810e+00, -2.449220e+00, -6.828130e+00, 8.765620e+00], [-9.906250e+00, -5.875000e+00, -5.224610e-01, -4.507810e+00, -5.640630e+00, -7.531250e+00, -5.179690e+00, -9.101560e+00, 6.574210e+00], [-7.796880e+00, -6.289060e+00, -8.486330e-01, -5.996090e+00, -4.039060e+00, -9.468750e+00, -2.496090e+00, -9.375000e+00, 4.539060e+00], [-1.418750e+01, -6.464840e+00, -1.474610e-01, -9.179680e+00, -3.437500e+00, -1.156250e+01, -2.490230e+00, -6.929680e+00, 7.511710e+00]]> : tensor<8x9xf16>
    return %cst : tensor<8x9xf16>
  }
  func.func private @cumsum(%arg0: tensor<8x9xf16>) -> tensor<8x9xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<f16>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<f16>
      stablehlo.return %2 : tensor<f16>
    }) : (tensor<8x9xf16>, tensor<f16>) -> tensor<8x9xf16>
    return %1 : tensor<8x9xf16>
  }
}
