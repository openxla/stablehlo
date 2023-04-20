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
    %0 = stablehlo.constant dense<[[-3.527830e-01, 3.957520e-01, -3.386720e+00, -2.726560e+00, -1.868160e+00, 3.339840e+00, 3.994140e+00, 4.613280e+00, -5.203130e+00], [-1.122070e+00, 3.394530e+00, -4.792970e+00, -2.671880e+00, 5.307620e-01, 1.410160e+00, 3.710940e+00, -1.458740e-01, -4.172360e-01], [-2.255860e+00, -6.847650e+00, 3.164060e+00, -5.035160e+00, -6.367190e-01, -1.781250e+00, -3.501950e+00, -3.023440e+00, 7.529300e-01], [6.157230e-01, -3.298340e-01, -7.485350e-01, -3.923830e+00, -2.188720e-01, 2.195310e+00, -1.279450e-02, -1.399410e+00, 2.707030e+00], [3.421880e+00, 1.920900e+00, 1.998050e+00, 9.633780e-01, 4.962160e-02, -1.998050e+00, 7.153320e-02, 7.318120e-02, 1.359380e+00], [-3.509770e+00, 5.968750e+00, -2.605470e+00, -7.289060e+00, 1.824950e-01, 2.785160e+00, 2.083980e+00, 4.072270e-01, -4.277340e+00], [3.001950e+00, 5.042970e+00, -1.151730e-01, -1.625000e+00, -3.940430e-01, -1.801760e+00, 6.113280e+00, 2.800780e+00, -3.103520e+00], [-3.117680e-01, 4.414060e+00, -1.665040e+00, -8.796870e+00, -1.251220e-01, 5.048830e-01, 4.710940e+00, -2.515630e+00, -7.781980e-02]]> : tensor<8x9xf16>
    return %0 : tensor<8x9xf16>
  }
  func.func private @expected() -> tensor<8x9xf16> {
    %0 = stablehlo.constant dense<[[-3.527830e-01, 3.957520e-01, -3.386720e+00, -2.726560e+00, -1.868160e+00, 3.339840e+00, 3.994140e+00, 4.613280e+00, -5.203130e+00], [2.758790e-02, 3.443360e+00, -3.167970e+00, -2.005860e+00, 6.176760e-01, 3.474610e+00, 4.554690e+00, 4.621090e+00, -4.089360e-01], [1.246340e-01, 3.443360e+00, 3.166020e+00, -1.958980e+00, 8.686520e-01, 3.480470e+00, 4.554690e+00, 4.621090e+00, 1.025390e+00], [1.093750e+00, 3.466800e+00, 3.185550e+00, -1.827150e+00, 1.159180e+00, 3.724610e+00, 4.566410e+00, 4.625000e+00, 2.878910e+00], [3.515630e+00, 3.660160e+00, 3.451170e+00, 1.023440e+00, 1.444340e+00, 3.728520e+00, 4.578130e+00, 4.636720e+00, 3.076170e+00], [3.517580e+00, 6.062500e+00, 3.453130e+00, 1.023440e+00, 1.693360e+00, 4.054690e+00, 4.656250e+00, 4.648440e+00, 3.078130e+00], [3.986330e+00, 6.371090e+00, 3.480470e+00, 1.091800e+00, 1.810550e+00, 4.058590e+00, 6.324210e+00, 4.792970e+00, 3.080080e+00], [3.998050e+00, 6.503900e+00, 3.488280e+00, 1.091800e+00, 1.944340e+00, 4.085940e+00, 6.503900e+00, 4.796880e+00, 3.121090e+00]]> : tensor<8x9xf16>
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
