// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf16>
    %1 = call @expected() : () -> tensor<8x9xf16>
    %2 = call @cumlogsumexp(%0) : (tensor<8x9xf16>) -> tensor<8x9xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf16>, tensor<8x9xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf16> {
    %0 = stablehlo.constant dense<[[6.535150e+00, 2.105470e+00, -3.685550e+00, 1.629640e-01, 1.678710e+00, 1.997070e-01, 1.563480e+00, 7.253900e+00, -6.835930e+00], [-5.679690e+00, 8.891600e-01, -4.281250e+00, 1.102540e+00, -5.640630e+00, 1.478520e+00, 4.609380e+00, -9.887690e-02, -1.129880e+00], [2.804690e+00, 5.132810e+00, 2.843750e+00, 9.189450e-01, -2.724610e+00, -8.968750e+00, -3.568360e+00, 5.351560e+00, 1.601560e+00], [-3.039550e-01, 1.848630e+00, -2.275390e-01, -1.256100e-01, -4.750000e+00, -2.619140e+00, -1.989260e+00, 9.942620e-02, 2.697270e+00], [-4.545900e-01, -5.082030e+00, 1.541990e+00, 3.015630e+00, 2.009770e+00, -3.076170e+00, -1.711910e+00, -1.548830e+00, -1.417970e+00], [6.179690e+00, 1.517580e+00, -7.699210e+00, -5.703130e-01, 1.064450e+00, -4.933590e+00, -2.970700e+00, 5.113280e+00, 4.238280e-01], [3.275390e+00, -8.569330e-01, 3.662110e-01, -1.558590e+00, -7.275390e-01, -2.837890e+00, 2.693360e+00, 5.102540e-01, -1.042970e+00], [-1.712890e+00, 4.496090e+00, -5.703130e+00, -1.238280e+00, -7.226560e-01, -1.152340e-01, -1.744140e+00, -3.335940e+00, 1.970700e+00]]> : tensor<8x9xf16>
    return %0 : tensor<8x9xf16>
  }
  func.func private @expected() -> tensor<8x9xf16> {
    %0 = stablehlo.constant dense<[[6.535150e+00, 2.105470e+00, -3.685550e+00, 1.629640e-01, 1.678710e+00, 1.997070e-01, 1.563480e+00, 7.253900e+00, -6.835930e+00], [6.535150e+00, 2.365230e+00, -3.246090e+00, 1.432620e+00, 1.679690e+00, 1.724610e+00, 4.656250e+00, 7.253900e+00, -1.126950e+00], [6.558590e+00, 5.195310e+00, 2.845700e+00, 1.901370e+00, 1.691410e+00, 1.724610e+00, 4.656250e+00, 7.394530e+00, 1.665040e+00], [6.558590e+00, 5.226560e+00, 2.890630e+00, 2.025390e+00, 1.693360e+00, 1.737300e+00, 4.656250e+00, 7.394530e+00, 3.001950e+00], [6.558590e+00, 5.226560e+00, 3.121090e+00, 3.332030e+00, 2.556640e+00, 1.745120e+00, 4.656250e+00, 7.394530e+00, 3.013670e+00], [7.078130e+00, 5.250000e+00, 3.121090e+00, 3.351560e+00, 2.759770e+00, 1.747070e+00, 4.660160e+00, 7.492180e+00, 3.085940e+00], [7.101560e+00, 5.253910e+00, 3.183590e+00, 3.359380e+00, 2.789060e+00, 1.756840e+00, 4.792970e+00, 7.492180e+00, 3.101560e+00], [7.101560e+00, 5.636710e+00, 3.183590e+00, 3.369140e+00, 2.820310e+00, 1.900390e+00, 4.789060e+00, 7.492180e+00, 3.382810e+00]]> : tensor<8x9xf16>
    return %0 : tensor<8x9xf16>
  }
  func.func private @cumlogsumexp(%arg0: tensor<8x9xf16>) -> tensor<8x9xf16> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<4x9xf16>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<4x9xf16>
    %2 = call @logaddexp(%0, %1) : (tensor<4x9xf16>, tensor<4x9xf16>) -> tensor<4x9xf16>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<2x9xf16>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<2x9xf16>
    %5 = call @logaddexp_0(%3, %4) : (tensor<2x9xf16>, tensor<2x9xf16>) -> tensor<2x9xf16>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %8 = call @logaddexp_1(%6, %7) : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf16>) -> tensor<0x9xf16>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<0x9xf16>
    %11 = call @logaddexp_2(%9, %10) : (tensor<0x9xf16>, tensor<0x9xf16>) -> tensor<0x9xf16>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xf16>, tensor<0x9xf16>) -> tensor<1x9xf16>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf16>, tensor<f16>) -> tensor<2x9xf16>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf16>, tensor<f16>) -> tensor<2x9xf16>
    %18 = stablehlo.add %15, %17 : tensor<2x9xf16>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf16>) -> tensor<1x9xf16>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<1x9xf16>
    %21 = call @logaddexp_3(%19, %20) : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xf16>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<1x9xf16>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<2x9xf16>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf16>, tensor<f16>) -> tensor<4x9xf16>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf16>, tensor<f16>) -> tensor<4x9xf16>
    %28 = stablehlo.add %25, %27 : tensor<4x9xf16>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf16>) -> tensor<3x9xf16>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<3x9xf16>
    %31 = call @logaddexp_4(%29, %30) : (tensor<3x9xf16>, tensor<3x9xf16>) -> tensor<3x9xf16>
    %32 = "stablehlo.slice"(%arg0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf16>) -> tensor<1x9xf16>
    %33 = stablehlo.concatenate %32, %31, dim = 0 : (tensor<1x9xf16>, tensor<3x9xf16>) -> tensor<4x9xf16>
    %34 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %35 = stablehlo.pad %33, %34, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xf16>, tensor<f16>) -> tensor<8x9xf16>
    %36 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %37 = stablehlo.pad %28, %36, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xf16>, tensor<f16>) -> tensor<8x9xf16>
    %38 = stablehlo.add %35, %37 : tensor<8x9xf16>
    return %38 : tensor<8x9xf16>
  }
  func.func private @logaddexp(%arg0: tensor<4x9xf16>, %arg1: tensor<4x9xf16>) -> tensor<4x9xf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<4x9xf16>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<4x9xf16>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<4x9xf16>, tensor<4x9xf16>) -> tensor<4x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<4x9xf16>
    %4 = stablehlo.abs %1 : tensor<4x9xf16>
    %5 = stablehlo.negate %4 : tensor<4x9xf16>
    %6 = stablehlo.exponential %5 : tensor<4x9xf16>
    %7 = stablehlo.log_plus_one %6 : tensor<4x9xf16>
    %8 = stablehlo.add %0, %7 : tensor<4x9xf16>
    %9 = stablehlo.select %2, %3, %8 : tensor<4x9xi1>, tensor<4x9xf16>
    return %9 : tensor<4x9xf16>
  }
  func.func private @logaddexp_0(%arg0: tensor<2x9xf16>, %arg1: tensor<2x9xf16>) -> tensor<2x9xf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<2x9xf16>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<2x9xf16>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<2x9xf16>, tensor<2x9xf16>) -> tensor<2x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<2x9xf16>
    %4 = stablehlo.abs %1 : tensor<2x9xf16>
    %5 = stablehlo.negate %4 : tensor<2x9xf16>
    %6 = stablehlo.exponential %5 : tensor<2x9xf16>
    %7 = stablehlo.log_plus_one %6 : tensor<2x9xf16>
    %8 = stablehlo.add %0, %7 : tensor<2x9xf16>
    %9 = stablehlo.select %2, %3, %8 : tensor<2x9xi1>, tensor<2x9xf16>
    return %9 : tensor<2x9xf16>
  }
  func.func private @logaddexp_1(%arg0: tensor<1x9xf16>, %arg1: tensor<1x9xf16>) -> tensor<1x9xf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<1x9xf16>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<1x9xf16>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<1x9xf16>
    %4 = stablehlo.abs %1 : tensor<1x9xf16>
    %5 = stablehlo.negate %4 : tensor<1x9xf16>
    %6 = stablehlo.exponential %5 : tensor<1x9xf16>
    %7 = stablehlo.log_plus_one %6 : tensor<1x9xf16>
    %8 = stablehlo.add %0, %7 : tensor<1x9xf16>
    %9 = stablehlo.select %2, %3, %8 : tensor<1x9xi1>, tensor<1x9xf16>
    return %9 : tensor<1x9xf16>
  }
  func.func private @logaddexp_2(%arg0: tensor<0x9xf16>, %arg1: tensor<0x9xf16>) -> tensor<0x9xf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<0x9xf16>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<0x9xf16>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<0x9xf16>, tensor<0x9xf16>) -> tensor<0x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<0x9xf16>
    %4 = stablehlo.abs %1 : tensor<0x9xf16>
    %5 = stablehlo.negate %4 : tensor<0x9xf16>
    %6 = stablehlo.exponential %5 : tensor<0x9xf16>
    %7 = stablehlo.log_plus_one %6 : tensor<0x9xf16>
    %8 = stablehlo.add %0, %7 : tensor<0x9xf16>
    %9 = stablehlo.select %2, %3, %8 : tensor<0x9xi1>, tensor<0x9xf16>
    return %9 : tensor<0x9xf16>
  }
  func.func private @logaddexp_3(%arg0: tensor<1x9xf16>, %arg1: tensor<1x9xf16>) -> tensor<1x9xf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<1x9xf16>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<1x9xf16>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<1x9xf16>, tensor<1x9xf16>) -> tensor<1x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<1x9xf16>
    %4 = stablehlo.abs %1 : tensor<1x9xf16>
    %5 = stablehlo.negate %4 : tensor<1x9xf16>
    %6 = stablehlo.exponential %5 : tensor<1x9xf16>
    %7 = stablehlo.log_plus_one %6 : tensor<1x9xf16>
    %8 = stablehlo.add %0, %7 : tensor<1x9xf16>
    %9 = stablehlo.select %2, %3, %8 : tensor<1x9xi1>, tensor<1x9xf16>
    return %9 : tensor<1x9xf16>
  }
  func.func private @logaddexp_4(%arg0: tensor<3x9xf16>, %arg1: tensor<3x9xf16>) -> tensor<3x9xf16> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<3x9xf16>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<3x9xf16>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<3x9xf16>, tensor<3x9xf16>) -> tensor<3x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<3x9xf16>
    %4 = stablehlo.abs %1 : tensor<3x9xf16>
    %5 = stablehlo.negate %4 : tensor<3x9xf16>
    %6 = stablehlo.exponential %5 : tensor<3x9xf16>
    %7 = stablehlo.log_plus_one %6 : tensor<3x9xf16>
    %8 = stablehlo.add %0, %7 : tensor<3x9xf16>
    %9 = stablehlo.select %2, %3, %8 : tensor<3x9xi1>, tensor<3x9xf16>
    return %9 : tensor<3x9xf16>
  }
}
