// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf32>
    %1 = call @expected() : () -> tensor<8x9xf32>
    %2 = call @cumlogsumexp(%0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf32>, tensor<8x9xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-1.20803022, -1.385580e+00, 0.727158963, -4.00244904, -3.17446899, 5.4945612, -1.8742578, -1.35144007, 0.891283512], [-0.641636729, -0.350393385, -3.07367778, 2.4045949, 0.400152981, 0.630957424, 4.64391232, 5.65796661, -1.62382448], [-2.25710225, 3.42243195, 2.78113413, 1.18097758, -4.43384457, -3.95816731, -2.98900723, -0.626821219, -2.11937857], [3.83401465, 6.97008085, 3.90120196, 1.67171693, -7.30881262, -1.14664567, 4.09895229, -4.05063486, 4.10961723], [0.20223856, -1.01655483, 0.52830106, -0.667989492, 9.489440e-02, -0.343670607, -4.201440e+00, -0.317248702, 1.868080e+00], [-0.594382346, 2.53058386, -2.37053418, 1.48209238, -3.37499404, 1.19866705, 1.50342929, 6.58710575, 1.66354382], [-3.12494302, 0.499667674, -5.74237251, -1.51614571, -4.68510151, 0.67633754, -0.81389153, 2.75878239, 0.091353789], [5.15596676, -5.40900278, 8.6190052, 1.7537272, -7.23260689, 9.32676792, 1.85914576, 2.20621037, -4.054600e+00]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-1.20803022, -1.385580e+00, 0.727158963, -4.00244904, -3.17446899, 5.4945612, -1.8742578, -1.35144007, 0.891283512], [-0.192110926, -0.0464737713, 0.749264895, 2.40624332, 0.427793473, 5.50225401, 4.64538765, 5.65886974, 0.969035148], [-0.072711952, 3.4531076, 2.90431619, 2.66373396, 0.435501486, 5.50233173, 4.64587116, 5.66073084, 1.0136013], [3.85392118, 6.99933815, 4.21530199, 2.97914886, 0.435934603, 5.50362635, 5.10249186, 5.6607914, 4.15385294], [3.87953758, 6.99966812, 4.2400403, 3.00488067, 0.973030328, 5.506510e+00, 5.10258293, 5.66332197, 4.25070333], [3.89087582, 7.01106071, 4.24138546, 3.20217514, 0.985879778, 5.51988268, 5.12956238, 6.92144298, 4.32324123], [3.89177299, 7.01254606, 4.24143171, 3.21106577, 0.98931837, 5.52773094, 5.13218212, 6.93688917, 4.33766174], [5.40475225, 7.01255035, 8.63148307, 3.42039847, 0.989586949, 9.34891414, 5.16937304, 6.9456706, 4.33788824]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cumlogsumexp(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %2 = call @logaddexp(%0, %1) : (tensor<4x9xf32>, tensor<4x9xf32>) -> tensor<4x9xf32>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %5 = call @logaddexp_0(%3, %4) : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2x9xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %8 = call @logaddexp_1(%6, %7) : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xf32>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf32>) -> tensor<0x9xf32>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<0x9xf32>
    %11 = call @logaddexp_2(%9, %10) : (tensor<0x9xf32>, tensor<0x9xf32>) -> tensor<0x9xf32>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xf32>, tensor<0x9xf32>) -> tensor<1x9xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %18 = stablehlo.add %15, %17 : tensor<2x9xf32>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %21 = call @logaddexp_3(%19, %20) : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xf32>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<2x9xf32>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %28 = stablehlo.add %25, %27 : tensor<4x9xf32>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<3x9xf32>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<3x9xf32>
    %31 = call @logaddexp_4(%29, %30) : (tensor<3x9xf32>, tensor<3x9xf32>) -> tensor<3x9xf32>
    %32 = "stablehlo.slice"(%arg0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<1x9xf32>
    %33 = stablehlo.concatenate %32, %31, dim = 0 : (tensor<1x9xf32>, tensor<3x9xf32>) -> tensor<4x9xf32>
    %34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %35 = stablehlo.pad %33, %34, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %37 = stablehlo.pad %28, %36, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %38 = stablehlo.add %35, %37 : tensor<8x9xf32>
    return %38 : tensor<8x9xf32>
  }
  func.func private @logaddexp(%arg0: tensor<4x9xf32>, %arg1: tensor<4x9xf32>) -> tensor<4x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<4x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<4x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<4x9xf32>, tensor<4x9xf32>) -> tensor<4x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<4x9xf32>
    %4 = stablehlo.abs %1 : tensor<4x9xf32>
    %5 = stablehlo.negate %4 : tensor<4x9xf32>
    %6 = stablehlo.exponential %5 : tensor<4x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<4x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<4x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<4x9xi1>, tensor<4x9xf32>
    return %9 : tensor<4x9xf32>
  }
  func.func private @logaddexp_0(%arg0: tensor<2x9xf32>, %arg1: tensor<2x9xf32>) -> tensor<2x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<2x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<2x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<2x9xf32>
    %4 = stablehlo.abs %1 : tensor<2x9xf32>
    %5 = stablehlo.negate %4 : tensor<2x9xf32>
    %6 = stablehlo.exponential %5 : tensor<2x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<2x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<2x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<2x9xi1>, tensor<2x9xf32>
    return %9 : tensor<2x9xf32>
  }
  func.func private @logaddexp_1(%arg0: tensor<1x9xf32>, %arg1: tensor<1x9xf32>) -> tensor<1x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<1x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<1x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<1x9xf32>
    %4 = stablehlo.abs %1 : tensor<1x9xf32>
    %5 = stablehlo.negate %4 : tensor<1x9xf32>
    %6 = stablehlo.exponential %5 : tensor<1x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<1x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<1x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<1x9xi1>, tensor<1x9xf32>
    return %9 : tensor<1x9xf32>
  }
  func.func private @logaddexp_2(%arg0: tensor<0x9xf32>, %arg1: tensor<0x9xf32>) -> tensor<0x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<0x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<0x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<0x9xf32>, tensor<0x9xf32>) -> tensor<0x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<0x9xf32>
    %4 = stablehlo.abs %1 : tensor<0x9xf32>
    %5 = stablehlo.negate %4 : tensor<0x9xf32>
    %6 = stablehlo.exponential %5 : tensor<0x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<0x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<0x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<0x9xi1>, tensor<0x9xf32>
    return %9 : tensor<0x9xf32>
  }
  func.func private @logaddexp_3(%arg0: tensor<1x9xf32>, %arg1: tensor<1x9xf32>) -> tensor<1x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<1x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<1x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<1x9xf32>
    %4 = stablehlo.abs %1 : tensor<1x9xf32>
    %5 = stablehlo.negate %4 : tensor<1x9xf32>
    %6 = stablehlo.exponential %5 : tensor<1x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<1x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<1x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<1x9xi1>, tensor<1x9xf32>
    return %9 : tensor<1x9xf32>
  }
  func.func private @logaddexp_4(%arg0: tensor<3x9xf32>, %arg1: tensor<3x9xf32>) -> tensor<3x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<3x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<3x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<3x9xf32>, tensor<3x9xf32>) -> tensor<3x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<3x9xf32>
    %4 = stablehlo.abs %1 : tensor<3x9xf32>
    %5 = stablehlo.negate %4 : tensor<3x9xf32>
    %6 = stablehlo.exponential %5 : tensor<3x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<3x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<3x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<3x9xi1>, tensor<3x9xf32>
    return %9 : tensor<3x9xf32>
  }
}
