// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xcomplex<f32>>
    %1 = call @expected() : () -> tensor<8x9xcomplex<f32>>
    %2 = call @cumprod(%0) : (tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xcomplex<f32>>, tensor<8x9xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-2.56053114,-0.365790695), (2.519740e+00,-1.64665461), (0.98595488,0.767968237), (-1.60500598,1.57992411), (-3.72184706,2.64282441), (-2.050540e+00,6.41372776), (-0.402819514,6.5522027), (0.287100315,3.09998441), (1.51857531,3.1667881)], [(-1.3109318,-4.12075663), (2.76577663,2.20780349), (0.652323961,4.40581465), (5.89588404,4.25839424), (0.212186635,-0.381584346), (0.106814995,0.149248809), (-2.72577119,-0.0750766769), (-1.87807274,-5.46468401), (-1.41774654,2.19836378)], [(-2.25933456,4.46958542), (-3.10455823,0.586564064), (-2.07422686,2.41041851), (3.17833066,3.81143665), (-3.6995821,1.89186323), (1.0993979,-2.03494978), (-4.454740e+00,-1.17979109), (-4.67660189,5.27406597), (-1.08287609,-1.95810318)], [(4.98991776,-3.61275649), (-5.29491377,-1.17384338), (-1.4724561,-0.47701022), (0.257514328,0.085031785), (-2.5044713,5.81217098), (3.00161386,-2.03347206), (-1.73332942,0.414024234), (-1.73081803,-0.911821305), (0.254062891,1.31767821)], [(-1.34202445,3.97389054), (1.12152982,-0.702221453), (-4.58332586,1.99798679), (-0.447755516,3.10611916), (2.39080739,1.61225343), (-7.26844597,1.14674258), (-1.06543791,1.97919452), (2.4069407,-2.58652163), (-2.10913825,-1.78470242)], [(-0.878138482,4.14506674), (-3.14547276,-2.63531566), (6.50250101,3.40588641), (2.4727931,-0.534484744), (-5.64046955,-3.81875205), (-3.24815297,-5.85002518), (6.88944149,3.23958397), (-0.811984419,3.57217073), (2.04303598,-1.56279719)], [(-5.32273579,-0.569616735), (2.5693891,1.12982786), (1.08178496,4.55053854), (-0.95915544,-0.401536673), (0.728645145,-0.331950575), (-1.78598118,3.1108706), (1.16192007,5.85507059), (-7.29371214,0.585811377), (3.77863955,1.16065836)], [(-0.623694956,0.443613112), (-2.66482782,-0.0616825782), (-2.05249453,-3.71875811), (-1.72700596,-0.41366896), (-1.66820729,-7.7828269), (1.19814622,2.83088684), (-1.45975268,5.43912745), (-1.39023495,1.58586681), (-8.325420e+00,-4.14648676)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @expected() -> tensor<8x9xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-2.56053114,-0.365790695), (2.519740e+00,-1.64665461), (0.98595488,0.767968237), (-1.60500598,1.57992411), (-3.72184706,2.64282441), (-2.050540e+00,6.41372776), (-0.402819514,6.5522027), (0.287100315,3.09998441), (1.51857531,3.1667881)], [(1.84934723,11.0308523), (10.6045284,1.00881195), (-2.7403636,4.8448987), (-16.1908684,2.48030114), (0.218734205,1.98097062), (-1.17626965,0.379041642), (1.58991146,-17.8295631), (16.4012394,-7.39090872), (-9.11470699,-1.15132189)], [(-53.4816322,-16.6565704), (-33.5141106,3.088320e+00), (-5.99409723,-16.6548424), (-60.9134445,-53.8272514), (-4.55695057,-6.91494846), (-0.521857679,2.81036735), (-28.1178017,77.5503082), (-37.7219276,121.065559), (7.61569118,19.0942764)], [(-327.045074,110.101189), (181.079514,22.9879265), (0.8815158,27.3827705), (-11.1090574,-19.0408669), (51.6036148,-9.16748619), (4.14838791,9.496820e+00), (16.6297073,-146.061691), (175.679947,-175.14679), (-23.2252464,14.886178)], [(1.37242496,-1447.39978), (219.228683,-101.376274), (-58.7506905,-123.742905), (64.1173401,-25.9804039), (138.154617,61.2804107), (-41.0427437,-6.427000e+01), (271.366577,188.533081), (-30.1697521,-875.967896), (75.5526505,10.0531464)], [(5998.36376,1276.70618), (-956.736389,-258.860474), (39.4278755,-1004.73657), (144.662781,-9.851390e+01), (-545.242249,-873.228576), (-242.667969,448.859833), (1258.79541,2178.00244), (3153.60425,603.500732), (170.06781,-97.5345306)], [(-31200.4727,-10212.3379), (-2165.76025,-1746.06067), (4614.74463,-907.490844), (-178.311035,36.4027328), (-687.15686,-455.280273), (-962.944396,-1556.56384), (-11289.7383,9901.00097), (-23355.0195,-2554.34351), (755.829224,-171.157196)], [(23989.9063,-7471.55615), (5663.67724,4786.54102), (-12846.4766,-15298.499), (323.00293,10.8940172), (-2397.04785,6107.52441), (3252.70801,-4590.97803), (-37372.5742,-75859.3438), (36519.8164,-33486.8164), (-7002.29736,-1709.08008)]]> : tensor<8x9xcomplex<f32>>
    return %0 : tensor<8x9xcomplex<f32>>
  }
  func.func private @cumprod(%arg0: tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %2 = stablehlo.multiply %0, %1 : tensor<4x9xcomplex<f32>>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %5 = stablehlo.multiply %3, %4 : tensor<2x9xcomplex<f32>>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %8 = stablehlo.multiply %6, %7 : tensor<1x9xcomplex<f32>>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xcomplex<f32>>) -> tensor<0x9xcomplex<f32>>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<0x9xcomplex<f32>>
    %11 = stablehlo.multiply %9, %10 : tensor<0x9xcomplex<f32>>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xcomplex<f32>>, tensor<0x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %14 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x9xcomplex<f32>>
    %16 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x9xcomplex<f32>>
    %18 = stablehlo.add %15, %17 : tensor<2x9xcomplex<f32>>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %21 = stablehlo.multiply %19, %20 : tensor<1x9xcomplex<f32>>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xcomplex<f32>>, tensor<1x9xcomplex<f32>>) -> tensor<2x9xcomplex<f32>>
    %24 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x9xcomplex<f32>>
    %26 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x9xcomplex<f32>>
    %28 = stablehlo.add %25, %27 : tensor<4x9xcomplex<f32>>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
    %31 = stablehlo.multiply %29, %30 : tensor<3x9xcomplex<f32>>
    %32 = "stablehlo.slice"(%arg0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xcomplex<f32>>) -> tensor<1x9xcomplex<f32>>
    %33 = stablehlo.concatenate %32, %31, dim = 0 : (tensor<1x9xcomplex<f32>>, tensor<3x9xcomplex<f32>>) -> tensor<4x9xcomplex<f32>>
    %34 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %35 = stablehlo.pad %33, %34, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<8x9xcomplex<f32>>
    %36 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %37 = stablehlo.pad %28, %36, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<8x9xcomplex<f32>>
    %38 = stablehlo.add %35, %37 : tensor<8x9xcomplex<f32>>
    return %38 : tensor<8x9xcomplex<f32>>
  }
}
